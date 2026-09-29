#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include "kernels.cuh" 

using namespace std;

int main(int argc, char* argv[]) {
    
    if (argc != 3) {
        cerr << "Usage: ./a2 <input_file> <algorithm>" << endl;
        return 1;
    }

    string input_file = argv[1];
    string algorithm = argv[2];

    ifstream infile(input_file);
    if (!infile) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    int n, k, t;
    infile >> n >> k >> t;

    vector<int> h_x(n), h_y(n), h_z(n);
    vector<uint8_t> h_I(n), h_out_I(n);

    for(int i = 0; i < n; i++) {
        int intensity;
        infile >> h_x[i] >> h_y[i] >> h_z[i] >> intensity;
        h_I[i] = static_cast<uint8_t>(intensity);
    }
    infile.close();

    string output_file = algorithm + ".txt";
    ofstream outfile(output_file.data());


    // auto start = chrono::high_resolution_clock::now();

    if (algorithm == "knn") {
        KNN_local_histogram_equalization_cuda(h_x.data(), h_y.data(), h_z.data(), h_I.data(), n, k, h_out_I.data());
    } 
    else if (algorithm == "approx_knn") {
        approx_knn_histogram_equalization_cuda(h_x.data(), h_y.data(), h_z.data(), h_I.data(), n, k, h_out_I.data());
    } 
    else if (algorithm == "kmeans") {
        k_mean_local_histogram_equalization_cuda(h_x.data(), h_y.data(), h_z.data(), h_I.data(), n, k, t, h_out_I.data());
    } 
    else {
        cerr << "Unknown algorithm!" << endl;
        return 1;
    }

    for(int i = 0; i < n; i++){
        outfile << h_x[i] << " " << h_y[i] << " " << h_z[i] << " " << (int)h_out_I[i] << "\n";
    }
    outfile.close();

    // auto end = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    // cout << algorithm << " Execution Time: " << duration.count() / 1000.0 << " s" << endl;

    return 0;
}