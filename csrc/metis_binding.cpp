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
    
    // Output arrays (preallocated NumPy array)
    py::array_t<int64_t> part_arr(nvtxs);
    idx_t* part_ptr = reinterpret_cast<idx_t*>(part_arr.mutable_data());
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
    
    return part_arr;
}

// Partition sparse matrix / graph to compute fill-reducing ordering
py::tuple metis_node_nd(
    py::array_t<int64_t> xadj,
    py::array_t<int64_t> adjncy,
    py::array_t<int64_t> vwgt) {
    
    py::buffer_info xadj_buf = xadj.request();
    py::buffer_info adjncy_buf = adjncy.request();
    
    if (xadj_buf.ndim != 1) {
        throw std::runtime_error("xadj must be 1-dimensional");
    }
    if (adjncy_buf.ndim != 1) {
        throw std::runtime_error("adjncy must be 1-dimensional");
    }
    
    idx_t nvtxs = xadj_buf.shape[0] - 1;
    
    idx_t* xadj_ptr = static_cast<idx_t*>(xadj_buf.ptr);
    idx_t* adjncy_ptr = static_cast<idx_t*>(adjncy_buf.ptr);
    idx_t* vwgt_ptr = nullptr;
    
    if (vwgt.size() > 0) {
        py::buffer_info vwgt_buf = vwgt.request();
        if (vwgt_buf.ndim != 1) {
            throw std::runtime_error("vwgt must be 1-dimensional");
        }
        if (vwgt_buf.shape[0] != nvtxs) {
            throw std::runtime_error("vwgt size must be equal to nvtxs");
        }
        vwgt_ptr = static_cast<idx_t*>(vwgt_buf.ptr);
    }
    
    py::array_t<int64_t> perm_arr(nvtxs);
    py::array_t<int64_t> iperm_arr(nvtxs);
    
    idx_t* perm_ptr = reinterpret_cast<idx_t*>(perm_arr.mutable_data());
    idx_t* iperm_ptr = reinterpret_cast<idx_t*>(iperm_arr.mutable_data());
    
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    
    int result = METIS_NodeND(
        &nvtxs,
        xadj_ptr,
        adjncy_ptr,
        vwgt_ptr,
        options,
        perm_ptr,
        iperm_ptr
    );
    
    if (result != METIS_OK) {
        if (result == METIS_ERROR_INPUT) {
            throw std::runtime_error("METIS error: Invalid input");
        } else if (result == METIS_ERROR_MEMORY) {
            throw std::runtime_error("METIS error: Memory allocation failed");
        } else {
            throw std::runtime_error("METIS error: Unknown error");
        }
    }
    
    return py::make_tuple(perm_arr, iperm_arr);
}

// Partition mesh using nodal connectivity
py::tuple metis_part_mesh_nodal(
    int64_t ne,
    int64_t nn,
    py::array_t<int64_t> eptr,
    py::array_t<int64_t> eind,
    py::array_t<int64_t> vwgt,
    py::array_t<int64_t> vsize,
    int64_t nparts) {
    
    py::buffer_info eptr_buf = eptr.request();
    py::buffer_info eind_buf = eind.request();
    
    if (eptr_buf.ndim != 1) {
        throw std::runtime_error("eptr must be 1-dimensional");
    }
    if (eind_buf.ndim != 1) {
        throw std::runtime_error("eind must be 1-dimensional");
    }
    if (eptr_buf.shape[0] != ne + 1) {
        throw std::runtime_error("eptr shape[0] must be ne + 1");
    }
    
    idx_t* eptr_ptr = static_cast<idx_t*>(eptr_buf.ptr);
    idx_t* eind_ptr = static_cast<idx_t*>(eind_buf.ptr);
    idx_t* vwgt_ptr = nullptr;
    idx_t* vsize_ptr = nullptr;
    
    if (vwgt.size() > 0) {
        py::buffer_info vwgt_buf = vwgt.request();
        if (vwgt_buf.ndim != 1) {
            throw std::runtime_error("vwgt must be 1-dimensional");
        }
        if (vwgt_buf.shape[0] != nn) {
            throw std::runtime_error("vwgt size must be equal to nn (number of nodes)");
        }
        vwgt_ptr = static_cast<idx_t*>(vwgt_buf.ptr);
    }
    
    if (vsize.size() > 0) {
        py::buffer_info vsize_buf = vsize.request();
        if (vsize_buf.ndim != 1) {
            throw std::runtime_error("vsize must be 1-dimensional");
        }
        if (vsize_buf.shape[0] != nn) {
            throw std::runtime_error("vsize size must be equal to nn (number of nodes)");
        }
        vsize_ptr = static_cast<idx_t*>(vsize_buf.ptr);
    }
    
    py::array_t<int64_t> epart_arr(ne);
    py::array_t<int64_t> npart_arr(nn);
    
    idx_t* epart_ptr = reinterpret_cast<idx_t*>(epart_arr.mutable_data());
    idx_t* npart_ptr = reinterpret_cast<idx_t*>(npart_arr.mutable_data());
    
    idx_t idx_ne = ne;
    idx_t idx_nn = nn;
    idx_t idx_nparts = nparts;
    idx_t objval = 0;
    
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    
    int result = METIS_PartMeshNodal(
        &idx_ne,
        &idx_nn,
        eptr_ptr,
        eind_ptr,
        vwgt_ptr,
        vsize_ptr,
        &idx_nparts,
        nullptr, // tpwgts (nullptr balances partition equally)
        options,
        &objval,
        epart_ptr,
        npart_ptr
    );
    
    if (result != METIS_OK) {
        if (result == METIS_ERROR_INPUT) {
            throw std::runtime_error("METIS error: Invalid input");
        } else if (result == METIS_ERROR_MEMORY) {
            throw std::runtime_error("METIS error: Memory allocation failed");
        } else {
            throw std::runtime_error("METIS error: Unknown error");
        }
    }
    
    return py::make_tuple(objval, epart_arr, npart_arr);
}

// Partition mesh using dual connectivity
py::tuple metis_part_mesh_dual(
    int64_t ne,
    int64_t nn,
    py::array_t<int64_t> eptr,
    py::array_t<int64_t> eind,
    py::array_t<int64_t> vwgt,
    py::array_t<int64_t> vsize,
    int64_t ncommon,
    int64_t nparts) {
    
    py::buffer_info eptr_buf = eptr.request();
    py::buffer_info eind_buf = eind.request();
    
    if (eptr_buf.ndim != 1) {
        throw std::runtime_error("eptr must be 1-dimensional");
    }
    if (eind_buf.ndim != 1) {
        throw std::runtime_error("eind must be 1-dimensional");
    }
    if (eptr_buf.shape[0] != ne + 1) {
        throw std::runtime_error("eptr shape[0] must be ne + 1");
    }
    
    idx_t* eptr_ptr = static_cast<idx_t*>(eptr_buf.ptr);
    idx_t* eind_ptr = static_cast<idx_t*>(eind_buf.ptr);
    idx_t* vwgt_ptr = nullptr;
    idx_t* vsize_ptr = nullptr;
    
    if (vwgt.size() > 0) {
        py::buffer_info vwgt_buf = vwgt.request();
        if (vwgt_buf.ndim != 1) {
            throw std::runtime_error("vwgt must be 1-dimensional");
        }
        if (vwgt_buf.shape[0] != ne) {
            throw std::runtime_error("vwgt size must be equal to ne (number of elements)");
        }
        vwgt_ptr = static_cast<idx_t*>(vwgt_buf.ptr);
    }
    
    if (vsize.size() > 0) {
        py::buffer_info vsize_buf = vsize.request();
        if (vsize_buf.ndim != 1) {
            throw std::runtime_error("vsize must be 1-dimensional");
        }
        if (vsize_buf.shape[0] != ne) {
            throw std::runtime_error("vsize size must be equal to ne (number of elements)");
        }
        vsize_ptr = static_cast<idx_t*>(vsize_buf.ptr);
    }
    
    py::array_t<int64_t> epart_arr(ne);
    py::array_t<int64_t> npart_arr(nn);
    
    idx_t* epart_ptr = reinterpret_cast<idx_t*>(epart_arr.mutable_data());
    idx_t* npart_ptr = reinterpret_cast<idx_t*>(npart_arr.mutable_data());
    
    idx_t idx_ne = ne;
    idx_t idx_nn = nn;
    idx_t idx_ncommon = ncommon;
    idx_t idx_nparts = nparts;
    idx_t objval = 0;
    
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    
    int result = METIS_PartMeshDual(
        &idx_ne,
        &idx_nn,
        eptr_ptr,
        eind_ptr,
        vwgt_ptr,
        vsize_ptr,
        &idx_ncommon,
        &idx_nparts,
        nullptr, // tpwgts (nullptr balances partition equally)
        options,
        &objval,
        epart_ptr,
        npart_ptr
    );
    
    if (result != METIS_OK) {
        if (result == METIS_ERROR_INPUT) {
            throw std::runtime_error("METIS error: Invalid input");
        } else if (result == METIS_ERROR_MEMORY) {
            throw std::runtime_error("METIS error: Memory allocation failed");
        } else {
            throw std::runtime_error("METIS error: Unknown error");
        }
    }
    
    return py::make_tuple(objval, epart_arr, npart_arr);
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
          
    m.def("node_nd", &metis_node_nd, "Compute fill-reducing ordering of a sparse matrix/graph using METIS",
          py::arg("xadj"),
          py::arg("adjncy"),
          py::arg("vwgt") = py::array_t<int64_t>());
          
    m.def("part_mesh_nodal", &metis_part_mesh_nodal, "Partition a mesh using nodal connectivity",
          py::arg("ne"),
          py::arg("nn"),
          py::arg("eptr"),
          py::arg("eind"),
          py::arg("vwgt") = py::array_t<int64_t>(),
          py::arg("vsize") = py::array_t<int64_t>(),
          py::arg("nparts") = 1);
          
    m.def("part_mesh_dual", &metis_part_mesh_dual, "Partition a mesh using dual connectivity",
          py::arg("ne"),
          py::arg("nn"),
          py::arg("eptr"),
          py::arg("eind"),
          py::arg("vwgt") = py::array_t<int64_t>(),
          py::arg("vsize") = py::array_t<int64_t>(),
          py::arg("ncommon") = 1,
          py::arg("nparts") = 1);
}

