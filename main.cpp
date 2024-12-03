#include <iostream>
#include <queue>
#include <cmath>
#include "LSH.h"
#include "reader.h"

using namespace std;

/*
    El paper utiliza la distancia euclidiana
*/

double euclideanDistance(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must have the same size.");
    }
    return (vec1 - vec2).norm();
}

/*
    Computar vecinos exactos
    Importante para el recall
*/
vector<pair<int, double>> findKNearestNeighbors(
    const Eigen::VectorXd& query,
    const vector<Eigen::VectorXd>& dataset,
    int K) {
    if (K <= 0) {
        throw invalid_argument("K must be greater than 0.");
    }

    using Neighbor = pair<double, int>;
    priority_queue<Neighbor, vector<Neighbor>, greater<>> minHeap;

    for (int i = 0; i < dataset.size(); ++i) {
        double distance = euclideanDistance(query, dataset[i]);
        minHeap.push({distance, i});
    }

    vector<pair<int, double>> kNearestNeighbors;
    for (int i = 0; i < K && !minHeap.empty(); ++i) {
        kNearestNeighbors.push_back({minHeap.top().second, minHeap.top().first});
        minHeap.pop();
    }

    return kNearestNeighbors;
}

int main() {
    
    /* 1. Cargamos el dataset y nuestro punto de consulta. */
    vector<Eigen::VectorXd> dataset = readFVECS("./movielens/movielens_base.fvecs");
    Eigen::VectorXd query = readFVECS("./movielens/movielens_query.fvecs")[0];

    /* 2. Configuramos nuestro LSH. */
    int K = 50;
    int L = 16;
    int d = dataset[0].size();
    double w = 5.0;
    LSH lsh(K, L, d, w);

    /* 3. Nuestro LSH indexa todos los puntos del dataset. */
    auto buckets = lsh.assign_to_buckets(dataset);

    /* 4. Hacemos una consulta directa a nuestro LSH. */
    vector<Eigen::VectorXd> candidates = lsh.query(query, buckets);

    /* 5. Mostramos los candidatos que nos dio nuestro LSH. */
    int i = 1;
    cout << "Number of candidates (LSH) = " << candidates.size() << endl;
    for (const Eigen::VectorXd& candidate: candidates) {
        cout << "   Euclidean distance between query and candidate " << i << " = " << euclideanDistance(query, candidate) << endl;
        i++;
    }
    /* 
        6. Calculamos los K = 10 vecinos mas cercanos exactos al punto de consulta.
        (Esto nos ayuda a comprobar si los candidatos de LSH son cercanos al punto de consulta). 
    */
    vector<pair<int, double>> knn = findKNearestNeighbors(query, dataset, 10);
    cout << "Exact 10 nearest neighbors" << endl;
    i = 1;
    for (const auto& [index, distance] : knn) {
        cout << "   Euclidean distance between query and neighbor " << i << " = " << distance << endl;
        i++;
    }

    return 0;
}