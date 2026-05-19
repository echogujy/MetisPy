#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <metis.h>

namespace py = pybind11;

// Partition graph using METIS
py::array_t<int64_t> metis_partition(
    py::array_t<int64_t> xadj,
    py::array_t<int64_t> adjncy,
    int64_t nparts,
    py::array_t<int64_t> vwgt,
    py::array_t<int64_t> adjwgt,
    int64_t ncon,
    std::string method) {
    
    // Get buffer info
    py::buffer_info xadj_buf = xadj.request();
    py::buffer_info adjncy_buf = adjncy.request();
    
    // Check dimensions
    if (xadj_buf.ndim != 1) {
        throw std::runtime_error("xadj must be 1-dimensional");
    }
    if (adjncy_buf.ndim != 1) {
        throw std::runtime_error("adjncy must be 1-dimensional");
    }
    
    // Get graph parameters
    idx_t nvtxs = xadj_buf.shape[0] - 1;
    idx_t nnz = adjncy_buf.shape[0];
    
    // Prepare METIS parameters
    idx_t* xadj_ptr = static_cast<idx_t*>(xadj_buf.ptr);
    idx_t* adjncy_ptr = static_cast<idx_t*>(adjncy_buf.ptr);
    idx_t* vwgt_ptr = nullptr;
    idx_t* adjwgt_ptr = nullptr;
    
    if (vwgt.size() > 0) {
        py::buffer_info vwgt_buf = vwgt.request();
        vwgt_ptr = static_cast<idx_t*>(vwgt_buf.ptr);
    }
    
    if (adjwgt.size() > 0) {
        py::buffer_info adjwgt_buf = adjwgt.request();
        adjwgt_ptr = static_cast<idx_t*>(adjwgt_buf.ptr);
    }
    
    // Output arrays
    std::vector<int64_t> part_vec(nvtxs);
    idx_t* part_ptr = reinterpret_cast<idx_t*>(part_vec.data());
    idx_t edgecut;
    
    // METIS options (use defaults)
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    
    int result;
    if (method == "kway") {
        result = METIS_PartGraphKway(
            &nvtxs,           // number of vertices
            &ncon,            // number of balancing constraints
            xadj_ptr,         // pointer to xadj array
            adjncy_ptr,       // pointer to adjncy array
            vwgt_ptr,         // vertex weights
            nullptr,          // vertex size (not used)
            adjwgt_ptr,       // edge weights
            &nparts,          // number of parts
            nullptr,          // target partition weights
            nullptr,          // ubvec (unbalance tolerance)
            options,          // options array
            &edgecut,         // output: edge cut
            part_ptr          // output: partition array
        );
    } else if (method == "recursive") {
        result = METIS_PartGraphRecursive(
            &nvtxs,           // number of vertices
            &ncon,            // number of balancing constraints
            xadj_ptr,         // pointer to xadj array
            adjncy_ptr,       // pointer to adjncy array
            vwgt_ptr,         // vertex weights
            nullptr,          // vertex size (not used)
            adjwgt_ptr,       // edge weights
            &nparts,          // number of parts
            nullptr,          // target partition weights
            nullptr,          // ubvec (unbalance tolerance)
            options,          // options array
            &edgecut,         // output: edge cut
            part_ptr          // output: partition array
        );
    } else {
        throw std::runtime_error("Method must be 'kway' or 'recursive'");
    }
    
    if (result != METIS_OK) {
        if (result == METIS_ERROR_INPUT) {
            throw std::runtime_error("METIS error: Invalid input");
        } else if (result == METIS_ERROR_MEMORY) {
            throw std::runtime_error("METIS error: Memory allocation failed");
        } else {
            throw std::runtime_error("METIS error: Unknown error");
        }
    }
    
    // Return as numpy array
    return py::array_t<int64_t>(
        py::buffer_info(
            part_vec.data(),
            sizeof(int64_t),
            py::format_descriptor<int64_t>::format(),
            1,
            {nvtxs},
            {sizeof(int64_t)}
        )
    );
}

// PyBind11 module
PYBIND11_MODULE(_metis, m) {
    m.doc() = "METIS graph partitioning bindings";
    m.def("partition", &metis_partition, "Partition graph using METIS",
          py::arg("xadj"),
          py::arg("adjncy"),
          py::arg("nparts"),
          py::arg("vwgt") = py::array_t<int64_t>(),
          py::arg("adjwgt") = py::array_t<int64_t>(),
          py::arg("ncon") = 1,
          py::arg("method") = "kway");
}
