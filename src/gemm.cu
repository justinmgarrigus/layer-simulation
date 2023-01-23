#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include <tools/util/host_matrix.h>

#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

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

int main(int argc, char **argv) {
    option const long_opts[] = {
        { "x", required_argument, nullptr, 'x' },
        { "w", required_argument, nullptr, 'w' },
        { nullptr, no_argument, nullptr, 0 },
    };

    Matrix w, x;

    int opt;
    while ((opt = getopt_long(argc, argv, "w:x:", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'w':
            w.load(optarg);
            break;
        case 'x':
            x.load(optarg);
            break;
        }
    }
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

    Gemm::launch(params);
    C.sync_host();

    Matrix gemm(w.rows(), x.cols());

    // Convert back to row-major
    for (size_t i = 0; i < gemm.rows(); ++i) {
        for (size_t j = 0; j < gemm.cols(); ++j) {
            gemm[i * gemm.cols() + j] = C.host_data()[j * gemm.rows() + i];
        }
    }

    gemm.save("bin/gemm.bin");
}
