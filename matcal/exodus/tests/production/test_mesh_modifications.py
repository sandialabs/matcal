import os
import numpy as np
import shutil
import copy 

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.exodus.mesh_modifications import ExodusHex8MeshExploder
from matcal.exodus.tests.utilities import test_support_files_dir, _open_mesh

class MeshExploderTests(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self.mesh_filename = "test_mesh_exploder_gold_mesh.g"
        shutil.copy(os.path.join(test_support_files_dir, self.mesh_filename), os.getcwd())
        self._num_x_nodes = 4
        self._num_y_nodes = 5
        self._num_z_nodes = 3
        self._num_eles = (self._num_x_nodes-1)*(self._num_y_nodes-1)*(self._num_z_nodes-1)

        self._ele_type = "HEX8"

    def test_boom_initialize_exploded_mesh(self):
        mesh_exploder = ExodusHex8MeshExploder(self.mesh_filename)
        mesh_exploder.boom()

        truth_connectivity, truth_xcoords, truth_ycoords, truth_zcoords = self._get_truth_mesh_coords_and_connectivity()

        exp_mesh = _open_mesh(self.mesh_filename)
        xcoords, ycoords, zcoords = exp_mesh.get_coords()
        block_1_connectivity = exp_mesh.get_elem_connectivity(1)[0]
        block_2_connectivity = exp_mesh.get_elem_connectivity(2)[0]
        block_names = exp_mesh.get_elem_blk_names()
        exp_mesh.close()

        from matcal.full_field.TwoDimensionalFieldGrid import NewToOldRemapper
        self.assert_close_arrays(xcoords, truth_xcoords)
        self.assert_close_arrays(ycoords, truth_ycoords)
        self.assert_close_arrays(zcoords, truth_zcoords)
        self.assert_close_arrays(block_1_connectivity, truth_connectivity[1])
        self.assert_close_arrays(block_2_connectivity, truth_connectivity[2])
        self.assertEqual(["block_main", "block_2_main"], block_names)

    def _get_truth_mesh_coords_and_connectivity(self):
        x_coords = np.linspace(-0.5, 0.5, self._num_x_nodes)
        y_coords = np.linspace(-1, 1, self._num_y_nodes)
        z_coords = np.linspace(-0.25, 0.25, self._num_z_nodes)

        x_mg_coords, y_mg_coords, z_mg_coords = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            
        connectivity = []
        all_node_xcoords = []
        all_node_ycoords = []
        all_node_zcoords = []
        first_block_made=False
        eles_built = 0
        connectivity_by_block = {}
        for z_ele in range(self._num_z_nodes-2, -1, -1):
            for x_ele in range(self._num_x_nodes-1):
                for y_ele in range(self._num_y_nodes-1):
                    all_node_xcoords.append(x_mg_coords[x_ele, y_ele, z_ele+1])
                    all_node_xcoords.append(x_mg_coords[x_ele, y_ele, z_ele])
                    all_node_xcoords.append(x_mg_coords[x_ele, y_ele+1, z_ele])
                    all_node_xcoords.append(x_mg_coords[x_ele, y_ele+1, z_ele+1])
                    all_node_xcoords.append(x_mg_coords[x_ele+1, y_ele, z_ele+1])
                    all_node_xcoords.append(x_mg_coords[x_ele+1, y_ele, z_ele])
                    all_node_xcoords.append(x_mg_coords[x_ele+1, y_ele+1, z_ele])
                    all_node_xcoords.append(x_mg_coords[x_ele+1, y_ele+1, z_ele+1])
                    
                    all_node_ycoords.append(y_mg_coords[x_ele, y_ele, z_ele+1])
                    all_node_ycoords.append(y_mg_coords[x_ele, y_ele, z_ele])
                    all_node_ycoords.append(y_mg_coords[x_ele, y_ele+1, z_ele])
                    all_node_ycoords.append(y_mg_coords[x_ele, y_ele+1, z_ele+1])                   
                    all_node_ycoords.append(y_mg_coords[x_ele+1, y_ele, z_ele+1])
                    all_node_ycoords.append(y_mg_coords[x_ele+1, y_ele, z_ele])
                    all_node_ycoords.append(y_mg_coords[x_ele+1, y_ele+1, z_ele])
                    all_node_ycoords.append(y_mg_coords[x_ele+1, y_ele+1, z_ele+1])
                    
                    all_node_zcoords.append(z_mg_coords[x_ele, y_ele, z_ele+1])
                    all_node_zcoords.append(z_mg_coords[x_ele, y_ele, z_ele])
                    all_node_zcoords.append(z_mg_coords[x_ele, y_ele+1, z_ele])
                    all_node_zcoords.append(z_mg_coords[x_ele, y_ele+1, z_ele+1])
                    all_node_zcoords.append(z_mg_coords[x_ele+1, y_ele, z_ele+1])
                    all_node_zcoords.append(z_mg_coords[x_ele+1, y_ele, z_ele])
                    all_node_zcoords.append(z_mg_coords[x_ele+1, y_ele+1, z_ele])
                    all_node_zcoords.append(z_mg_coords[x_ele+1, y_ele+1, z_ele+1])

                    eles_built+=1
                    connectivity+=list(range((eles_built-1)*8, eles_built*8))

                    if eles_built >= self._num_eles/2 and not first_block_made:
                        first_block_made=True
                        connectivity = np.array(connectivity) + 1
                        connectivity_by_block[1] = copy.deepcopy(connectivity)
                        connectivity = []
                        first_block_eles_built = eles_built

        connectivity = np.array(connectivity) + 1
        connectivity_by_block[2] = copy.deepcopy(connectivity)
        return connectivity_by_block, all_node_xcoords, all_node_ycoords, all_node_zcoords

    def test_boom_copy_time_step_data_exploded_mesh(self):
        test_mesh = self._write_nodal_data_to_file(self.mesh_filename)
        node_variable_names = test_mesh.get_node_variable_names()
        test_mesh.close()
        mesh_exploder = ExodusHex8MeshExploder(self.mesh_filename)
        mesh_exploder.boom()
        exp_mesh = _open_mesh(self.mesh_filename)
        x_coords, y_coords, z_coords = exp_mesh.get_coords()
        times = exp_mesh.get_times()
        for time_step, time in enumerate(times):
            for node_var_name in node_variable_names:
                node_var_values = exp_mesh.get_node_variable_values(node_var_name, time_step+1)
                self.assert_close_arrays(node_var_values, node_var_funcs[node_var_name](x_coords, y_coords, time))
        exp_mesh.close()

    def _write_nodal_data_to_file(self, filename):
        test_mesh = _open_mesh(filename, open_mode="a")
        node_variable_names = ["displacement_x", "displacement_y", "T"]


        test_mesh = self._initialize_node_variables(test_mesh, node_variable_names)

        times = np.linspace(0, 20, 10)
        for time_step, time in enumerate(times):
            time_step += 1
            test_mesh.put_time(time_step, time)
            for node_variable_name in node_variable_names:
                x_coords, y_coords, z_coords = test_mesh.get_coords()
                new_mesh_node_variable_values = node_var_funcs[node_variable_name](x_coords, y_coords, time)
                test_mesh.put_node_variable_values(node_variable_name, time_step, new_mesh_node_variable_values)

        return test_mesh

    def _initialize_node_variables(self, mesh_object, node_variable_names):
        mesh_object.set_node_variable_number(len(node_variable_names))
        for node_variable_index, node_variable_name in enumerate(node_variable_names):
                mesh_object.put_node_variable_name(node_variable_name, node_variable_index+1)
        return mesh_object

    def test_boom_copy_node_set_info(self):
        test_mesh = self._write_nodal_data_to_file(self.mesh_filename)
        test_mesh.close()
        mesh_exploder = ExodusHex8MeshExploder(self.mesh_filename)
        mesh_exploder.boom()
        exp_mesh = _open_mesh(self.mesh_filename)
        
        gold_ids = [1,200]
        self.assert_close_arrays(gold_ids, exp_mesh.get_node_set_ids())
       
        exp_mesh_nset_nodes = {}
        truth_nset_names = ["nodeset1", "nodeset200"]
        for nodeset_index, nodeset_id in enumerate(exp_mesh.get_node_set_ids()):
            nset_name = exp_mesh.get_node_set_name(nodeset_id)
            self.assertEqual(truth_nset_names[nodeset_index], nset_name)
            nset_nodes = exp_mesh.get_node_set_nodes(nodeset_id)
            exp_mesh_nset_nodes[nodeset_index] = nset_nodes

        x_coords, y_coords, z_coords = exp_mesh.get_coords()
        
        exp_mesh_node_id_map = exp_mesh.get_node_id_map()
        nset1_truth_indices = np.where(z_coords == 0.25)[0]
        nset1_truth_ids = exp_mesh_node_id_map[nset1_truth_indices]
        for nset1_truth_id in nset1_truth_ids:
            self.assertTrue(nset1_truth_id in exp_mesh_nset_nodes[0])
        for nset1_id in exp_mesh_nset_nodes[0]:
            self.assertTrue(nset1_id in list(nset1_truth_ids))

        nset2_truth_indices = np.where(z_coords == -0.25)[0]
        nset2_truth_ids = exp_mesh_node_id_map[nset2_truth_indices]
        for nset2_truth_id in nset2_truth_ids:
            self.assertTrue(nset2_truth_id in exp_mesh_nset_nodes[1])
        for nset2_id in exp_mesh_nset_nodes[1]:
            self.assertTrue(nset2_id in nset2_truth_ids)

        exp_mesh.close()

def displacement_x(x_coords, y_coords, time):
    return 2*x_coords/(y_coords+2)*(time/10)**2

def displacement_y(x_coords, y_coords, time):
    return (x_coords*x_coords-y_coords) + 3*time/10

def temperature(x_coords, y_coords, time):
    return (x_coords*x_coords+y_coords*y_coords)*(time/10+1)

node_var_funcs = {"displacement_x": displacement_x, 
                  "displacement_y": displacement_y, 
                  "T": temperature}
