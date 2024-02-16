#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include <tools/util/host_matrix.h>

#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <cmath>

class Matrix {
public:
    Matrix() = default;
    Matrix(uint32_t rows, uint32_t cols) : m_rows{rows}, m_cols{cols} {
        m_data = new float[rows * cols];
    }
    Matrix(Matrix const &) = delete;

    ~Matrix() {
        delete[] m_data;
    }

    uint32_t rows() const {
        return m_rows;
    }

    uint32_t cols() const {
        return m_cols;
    }

    void load(std::string const &fpath);
    void save(std::string const &fpath);

    Matrix &operator=(Matrix const &) = delete;

    float &operator[](size_t index) {
        return m_data[index];
    }
    
    float const &operator[](size_t index) const {
        return m_data[index];
    }

private:
    float *m_data = nullptr;
    uint32_t m_rows;
    uint32_t m_cols;
};

void Matrix::load(std::string const &fpath) {
    std::ifstream ifs(fpath, std::ios::binary);
    if (!ifs) {
        throw std::system_error(
            errno,
            std::system_category(),
            "Failed to open \"" + fpath + "\""
        );
    }

    ifs.read(reinterpret_cast<char *>(&m_rows), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(&m_cols), sizeof(uint32_t));

    auto fsize = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    fsize = ifs.tellg() - fsize;

    ifs.clear();
    ifs.seekg(sizeof(uint32_t) * 2, std::ios::beg);

    auto count = fsize / sizeof(float);

    m_data = new float[count];

    for (size_t i = 0; i < count; ++i) {
        float buf;
        ifs.read(reinterpret_cast<char *>(&buf), sizeof(float));
        m_data[i] = buf;
    }
}

void Matrix::save(std::string const &fpath) {
    std::ofstream ofs(fpath, std::ios::binary);
    if (!ofs) {
        throw std::system_error(
            errno,
            std::system_category(),
            "Failed to open for write \"" + fpath + "\""
        );
    }

    ofs.write(reinterpret_cast<char *>(&m_rows), sizeof(uint32_t));
    ofs.write(reinterpret_cast<char *>(&m_cols), sizeof(uint32_t));

    ofs.write(reinterpret_cast<char *>(m_data), sizeof(float) * rows() * cols());

    ofs.close();
}

using WmmaGemmTraits = cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    float,
    cutlass::gemm::LinearScaling<float>,
    float,
    cutlass::gemm::WmmaGemmAccumulatorsPerWarp<
        cutlass::Shape<32, 16, 16>
    >::Shape,
    cutlass::Shape<16, 16, 16>
>;

using Gemm = cutlass::gemm::Gemm<WmmaGemmTraits>;
typename Gemm::Params params;

// Reads specific values from the file, assigning default values when the file
// does not exist.
void read_extern_values(int** data_ptr) {
    FILE *extern_vars_file = fopen("EXTERN_VARS.temp", "r");
    if (extern_vars_file == NULL) {
        *data_ptr = 0; 
    }
    else {
        assert(fscanf(extern_vars_file, "data_ptr=%p\n", data_ptr) != 0); 
        fclose(extern_vars_file);
    }  

    printf("UNT (SRC): value received, %p\n", *data_ptr);
}

// Writes specific values to the file.
void write_extern_values(
    char* B_start_ptr, 
    char* B_end_ptr, 
    int conv_batch_size, 
    int conv_input_channels, 
    int conv_input_rows, 
    int conv_input_cols, 
    int conv_filter_rows, 
    int conv_filter_cols, 
    int conv_stride_rows, 
    int conv_stride_cols, 
    int conv_padding_rows, 
    int conv_padding_cols, 
    int conv_output_channels) 
{
    FILE *extern_vars_file = fopen("EXTERN_VARS.temp", "w"); 
    fprintf(extern_vars_file, "B_start_ptr=%p\n", B_start_ptr);
    fprintf(extern_vars_file, "B_end_ptr=%p\n", B_end_ptr); 
    fprintf(extern_vars_file, "conv_batch_size=%d\n", conv_batch_size);
    fprintf(extern_vars_file, "conv_input_channels=%d\n", conv_input_channels);
    fprintf(extern_vars_file, "conv_input_rows=%d\n", conv_input_rows);
    fprintf(extern_vars_file, "conv_input_cols=%d\n", conv_input_cols);
    fprintf(extern_vars_file, "conv_filter_rows=%d\n", conv_filter_rows);
    fprintf(extern_vars_file, "conv_filter_cols=%d\n", conv_filter_cols);
    fprintf(extern_vars_file, "conv_stride_rows=%d\n", conv_stride_rows);
    fprintf(extern_vars_file, "conv_stride_cols=%d\n", conv_stride_cols);
    fprintf(extern_vars_file, "conv_padding_rows=%d\n", conv_padding_rows);
    fprintf(extern_vars_file, "conv_padding_cols=%d\n", conv_padding_cols);
    fprintf(extern_vars_file, "conv_output_channels=%d\n", 
        conv_output_channels);
    printf("UNT (SRC): B_start_ptr written, %p\n", B_start_ptr);
    printf("UNT (SRC): B_end_ptr written, %p\n", B_end_ptr);
    printf("UNT (SRC): conv_batch_size written, %d\n", conv_batch_size); 
    printf("UNT (SRC): conv_input_channels written, %d\n", 
        conv_input_channels); 
    printf("UNT (SRC): conv_input_rows written, %d\n", conv_input_rows); 
    printf("UNT (SRC): conv_input_cols written, %d\n", conv_input_cols); 
    printf("UNT (SRC): conv_filter_rows written, %d\n", conv_filter_rows); 
    printf("UNT (SRC): conv_filter_cols written, %d\n", conv_filter_cols); 
    printf("UNT (SRC): conv_stride_rows written, %d\n", conv_stride_rows); 
    printf("UNT (SRC): conv_stride_cols written, %d\n", conv_stride_cols); 
    printf("UNT (SRC): conv_padding_rows written, %d\n", conv_padding_rows); 
    printf("UNT (SRC): conv_padding_cols written, %d\n", conv_padding_cols); 
    printf("UNT (SRC): conv_output_channels written, %d\n", 
        conv_output_channels); 
    fclose(extern_vars_file);
}

int main(int argc, char **argv) {
    option const long_opts[] = {
        { "x", required_argument, nullptr, 'x' },
        { "w", required_argument, nullptr, 'w' },
        { "o", required_argument, nullptr, 'o' }, 
        { "c", no_argument, nullptr, 'c' }, // compare "x" and "w"
        { nullptr, no_argument, nullptr, 0 },
    };

    Matrix w, x;
    std::string output_file_dir("bin/gemm.bin"); 

    // 0 if we do NOT compare matrices (standard gemm), 1 if we DO. 
    int compare_matrices = 0;

    int opt;
    while ((opt = getopt_long(argc, argv, "w:x:o:c:", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'w':
            w.load(optarg);
            break;
        case 'x':
            x.load(optarg);
            break;
        case 'o': 
            output_file_dir = std::string(optarg);
            break; 
        case 'c':
            compare_matrices = 1; 
            break; 
        }
    }
    
    if (compare_matrices) {
        assert(w.rows() == x.rows() && w.cols() == x.cols()); 
        
        float distance = 0;
        for (int i = 0; i < w.rows() * w.cols(); i++)
            distance += std::abs(w[i] - x[i]); 
        
        fprintf(stdout, "%f\n", distance); 
    }
    else {
        assert(w.cols() == x.rows());
    
        cutlass::HostMatrixRowMajor<cutlass::half_t> A(cutlass::MatrixCoord(w.rows(), w.cols()));
        for (size_t i = 0; i < w.rows() * w.cols(); ++i) {
            A.host_data()[i] = cutlass::half_t::convert(w[i]);
        }
        A.sync_device();
    
        cutlass::HostMatrixRowMajor<cutlass::half_t> B(cutlass::MatrixCoord(x.rows(), x.cols()));
        for (size_t i = 0; i < x.rows() * x.cols(); ++i) {
            B.host_data()[i] = cutlass::half_t::convert(x[i]);
        }
        B.sync_device();
    
        cutlass::HostMatrix<float> C(cutlass::MatrixCoord(w.rows(), x.cols()));
    
        params.initialize(
            w.rows(),
            x.cols(),
            w.cols(),
            1.0f,
            A.device_data(),
            A.leading_dim(),
            B.device_data(),
            B.leading_dim(),
            0.0f,
            C.device_data(),
            C.leading_dim(),
            C.device_data(),
            C.leading_dim()
        );

        half *B_d = B.device_data();

        // The extern_vars file shouldn't initially exist. It may exist if 
        // it wasn't cleaned from the previous simulator session. 
        remove("EXTERN_VARS.temp"); 

        // Pass values into the simulator to indicate the status of the 
        // convolution kernel. Most of the parameters will be hard-coded into
        // this (TODO). 
        char *B_start_ptr = (char*)B.device_data(); 
        char *B_end_ptr = B_start_ptr + sizeof(half) * x.rows() * x.cols(); 
        int conv_batch_size = 1; 
        int conv_input_channels = 3; 
        int conv_input_rows = 224; 
        int conv_input_cols = 224; 
        int conv_filter_rows = 11; 
        int conv_filter_cols = 11; 
        int conv_stride_rows = 4; 
        int conv_stride_cols = 4; 
        int conv_padding_rows = 2; 
        int conv_padding_cols = 2; 
        int conv_output_channels = 64; 
        write_extern_values(
            B_start_ptr,
            B_end_ptr, 
            conv_batch_size, 
            conv_input_channels, 
            conv_input_rows,
            conv_input_cols, 
            conv_filter_rows,
            conv_filter_cols, 
            conv_stride_rows, 
            conv_stride_cols, 
            conv_padding_rows,
            conv_padding_cols, 
            conv_output_channels
        );
    
        Gemm::launch(params);
        C.sync_host();
    
        Matrix gemm(w.rows(), x.cols());
    
        // Convert back to row-major
        for (size_t i = 0; i < gemm.rows(); ++i) {
            for (size_t j = 0; j < gemm.cols(); ++j) {
                gemm[i * gemm.cols() + j] = C.host_data()[j * gemm.rows() + i];
            }
        }
        
        gemm.save(output_file_dir);
    }
}
