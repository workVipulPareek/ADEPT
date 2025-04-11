import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity
from geopy.distance import geodesic
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

class EmergencyClusterAnalyzer:
    def __init__(self, data_path="/fab3/btech/2022/vipul.pareek22b/Project/data/emergency_service_data.csv", 
                 num_ambulances=None, n_clusters=None, 
                 output_dir="project_RL", imperfection_ratio=0.1):
        self.data_path = data_path
        self.df = None
        self.cluster_labels = None
        self.scaler = RobustScaler()
        self.spatial_scaler = RobustScaler()
        self.feature_columns = None
        self.allocation = None
        self.cluster_centers = None
        self.num_ambulances = num_ambulances
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.imperfection_ratio = max(0, min(1, imperfection_ratio))
        
        # Create necessary directories
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # File paths
        self.cluster_image_path = os.path.join(output_dir, "clusters.png")
        self.static_allocation_path = os.path.join(output_dir, "data", "static_allocation.npy")
        self.emergency_data_path = os.path.join(output_dir, "data", "emergency_service_data.csv")
        self.rl_data_path = os.path.join(output_dir, "data", "rl_environment_data.json")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            required_columns = ['timestamp', 'latitude', 'longitude']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError("Data must contain timestamp, latitude, and longitude columns")
            
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['is_rush_hour'] = self.df['hour'].isin([7, 8, 16, 17]).astype(int)
            self.df.to_csv(self.emergency_data_path, index=False)
            logging.info(f"Data processed and saved to {self.emergency_data_path}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        coords = self.df[['latitude', 'longitude']].values
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(coords)
            score = silhouette_score(coords, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                
        logging.info(f"Optimal clusters found: {best_k} (silhouette score: {best_score:.2f})")
        return best_k

    def calculate_optimal_ambulances(self):
        """Calculate optimal number of ambulances based on incident density"""
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_clusters()
        
        cluster_counts = self.df.groupby('cluster').size()
        incidents_per_hour = cluster_counts / (24 * 30)  # Assuming 1 month of data
        required_ambulances = np.ceil(incidents_per_hour * 0.5).sum()  # 0.5 hours per incident
        return max(5, int(required_ambulances))  # Minimum 5 ambulances

    def perform_clustering(self):
        """Perform clustering and generate all required outputs"""
        if self.df is None:
            self.load_data()
        
        # Determine number of clusters
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_clusters()
        
        # Add imperfection to coordinates before clustering
        if self.imperfection_ratio > 0:
            noise_scale = 0.01 * self.imperfection_ratio
            self.df['latitude'] += np.random.normal(0, noise_scale, len(self.df))
            self.df['longitude'] += np.random.normal(0, noise_scale, len(self.df))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(self.df[['latitude', 'longitude']])
        self.cluster_centers = kmeans.cluster_centers_
        
        # Determine number of ambulances
        if self.num_ambulances is None:
            self.num_ambulances = self.calculate_optimal_ambulances()
        
        # Create static allocation proportional to cluster sizes
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        static_allocation = np.floor(cluster_counts / cluster_counts.sum() * self.num_ambulances).astype(int)
        
        # Distribute remaining ambulances
        remaining = self.num_ambulances - static_allocation.sum()
        if remaining > 0:
            static_allocation[-remaining:] += 1
        
        # Save static allocation
        np.save(self.static_allocation_path, static_allocation.values)
        logging.info(f"Static allocation saved to {self.static_allocation_path}")
        
        # Calculate incident rates (incidents per hour)
        total_hours = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 3600
        incident_rates = cluster_counts / total_hours
        
        # Create travel times matrix with some imperfection
        travel_times = np.zeros((self.num_ambulances, self.n_clusters))
        for i in range(self.num_ambulances):
            for j in range(self.n_clusters):
                # Base travel time + imperfection
                base_time = np.random.uniform(5, 15)
                imperfection = np.random.normal(0, 2 * self.imperfection_ratio)
                travel_times[i,j] = max(1, base_time + imperfection)
        
        # Prepare RL environment data
        rl_environment_data = {
            'n_clusters': int(self.n_clusters),
            'n_ambulances': int(self.num_ambulances),
            'incident_rates': incident_rates.tolist(),
            'travel_times': travel_times.tolist(),
            'imperfection_ratio': float(self.imperfection_ratio)
        }
        
        with open(self.rl_data_path, 'w') as f:
            json.dump(rl_environment_data, f, indent=2)
        logging.info(f"RL environment data saved to {self.rl_data_path}")
        
        # Visualize clusters
        self._visualize_clusters()
        
        return static_allocation.values

    def _visualize_clusters(self):
        """Visualize and save cluster plot"""
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='longitude',
            y='latitude',
            hue='cluster',
            data=self.df,
            palette='viridis',
            alpha=0.6
        )
        plt.scatter(
            self.cluster_centers[:, 1],
            self.cluster_centers[:, 0],
            marker='X',
            s=200,
            c='red',
            label='Cluster Centers'
        )
        plt.title(f'Emergency Incident Clusters (Imperfection: {self.imperfection_ratio})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.savefig(self.cluster_image_path)
        plt.close()
        logging.info(f"Cluster visualization saved to {self.cluster_image_path}")

    def run_analysis(self):
        """Run complete analysis pipeline"""
        try:
            self.load_data()
            static_allocation = self.perform_clustering()
            
            logging.info("Analysis completed successfully")
            print("\nGenerated files:")
            print(f"- Emergency data: {self.emergency_data_path}")
            print(f"- Cluster visualization: {self.cluster_image_path}")
            print(f"- Static allocation: {self.static_allocation_path}")
            print(f"- RL environment data: {self.rl_data_path}")
            
            print("\nConfiguration:")
            print(f"- Number of clusters: {self.n_clusters}")
            print(f"- Number of ambulances: {self.num_ambulances}")
            print(f"- Imperfection ratio: {self.imperfection_ratio}")
            print(f"\nStatic allocation: {static_allocation}")
            
            return static_allocation
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage with different parameters
    analyzer = EmergencyClusterAnalyzer(
        data_path="/fab3/btech/2022/vipul.pareek22b/Project/data/emergency_service_data.csv",
        num_ambulances=10,  # Let the system calculate optimal number
        n_clusters=5,  # Let the system determine optimal clusters
        output_dir="project_RL",
        imperfection_ratio=0.2  # Set imperfection level (0-1)
    )
    result = analyzer.run_analysis()