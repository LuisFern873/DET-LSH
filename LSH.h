#include "Eigen/Dense"
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include <iostream>
#include <set> 

/*
    𝐾: dimensión de cada espacio proyectado
    𝐿: número de espacios proyectados
    d: dimensión del espacio original
    w: ancho de los buckets
    H: hash functions (𝐾*𝐿)
*/

using namespace std;

class LSH {

private:
    int K, L, d;
    double w;
    vector<vector<pair<Eigen::VectorXd, double>>> H;
    random_device rd;
    mt19937 gen;
    normal_distribution<> dis;
    uniform_real_distribution<> uniform_dis;

public:
    LSH(int K, int L, int d, double w)
        : K(K), L(L), d(d), w(w), gen(rd()), dis(0.0, 1.0), uniform_dis(0.0, w) {
        H = generate_hash_functions();
    }

    struct VectorHash {
        std::size_t operator()(const std::vector<int>& vec) const {
            std::size_t hash = 0;
            for (int num : vec) {
                hash ^= std::hash<int>()(num) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    struct Comparator {
        bool operator()(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const {
            return a.isApprox(b);
        }
    };

    vector<vector<pair<Eigen::VectorXd, double>>> generate_hash_functions() {
        vector<vector<pair<Eigen::VectorXd, double>>> H(L, vector<pair<Eigen::VectorXd, double>>(K));
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < K; ++j) {
                Eigen::VectorXd a = Eigen::VectorXd::NullaryExpr(d, [&]() { return dis(gen); });

                double b = uniform_dis(gen);
                H[i][j] = make_pair(a, b);
            }
        }
        return H;
    }

    vector<int> project_point(const Eigen::VectorXd& point, int space_index) {
        vector<int> hashes(K);
        for (int j = 0; j < K; ++j) {
            const auto& [a, b] = H[space_index][j];
            double h = floor((a.dot(point) + b) / w);
            hashes[j] = static_cast<int>(h);
        }
        return hashes;
    }

    vector<vector<vector<int>>> project_dataset(const vector<Eigen::VectorXd>& dataset) {
        int n = dataset.size();
        vector<vector<vector<int>>> projected_points(L, vector<vector<int>>(n, vector<int>(K)));
        for (int i = 0; i < L; ++i) {
            for (int idx = 0; idx < n; ++idx) {
                projected_points[i][idx] = project_point(dataset[idx], i);
            }
        }
        return projected_points;
    }


    vector<unordered_map<vector<int>, vector<Eigen::VectorXd>, VectorHash>> assign_to_buckets(const vector<Eigen::VectorXd>& dataset) {
        // Cada bucket es un unordered_map donde la llave es un vector<int> y el valor es un vector de Eigen::VectorXd.
        vector<unordered_map<vector<int>, vector<Eigen::VectorXd>, VectorHash>> buckets(L);

        for (const auto& point : dataset) {
            for (int space_index = 0; space_index < L; ++space_index) {
                // Proyectamos el punto al espacio actual
                vector<int> bucket_key = project_point(point, space_index);
                // Asignamos el punto al bucket correspondiente
                buckets[space_index][bucket_key].push_back(point);
            }
        }

        return buckets;
    }

    vector<Eigen::VectorXd> query(const Eigen::VectorXd& query_point, const vector<unordered_map<vector<int>, vector<Eigen::VectorXd>, VectorHash>>& buckets) {
        
        set<Eigen::VectorXd, Comparator> candidates;

        for (int space_index = 0; space_index < L; ++space_index) {
            
            vector<int> bucket_key = project_point(query_point, space_index);

            // Si el bucket existe, añadimos los candidatos
            auto it = buckets[space_index].find(bucket_key);
            if (it != buckets[space_index].end()) {
                candidates.insert(it->second.begin(), it->second.end());
            }
        }

        return vector<Eigen::VectorXd>(candidates.begin(), candidates.end());
    }
};