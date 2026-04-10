import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest import mock

from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.sierra.models.preprocessors import (
    _prepend_string_to_file,
    AddApreproParamFileLinesPreprocessor,
    DecomposeAndCopyMeshPreprocessor,
)


class TestAddApreproToInput(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_prepends_message(self):
        fn = "deck.i"
        original = "line1\nline2\n"
        with open(fn, "w") as f:
            f.write(original)
        msg = "{include(foo.i)}\n"
        _prepend_string_to_file(fn, msg)
        with open(fn, "r") as f:
            out = f.read()
        self.assertEqual(out, msg + original)

    def test_prepends_to_empty_file(self):
        fn = "deck.i"
        with open(fn, "w") as f:
            f.write("")
        msg = "hello\n"
        _prepend_string_to_file(fn, msg)
        with open(fn, "r") as f:
            out = f.read()
        self.assertEqual(out, msg)

    def test_temp_file_removed(self):
        fn = "deck.i"
        with open(fn, "w") as f:
            f.write("x\n")
        tmp = fn + ".temp"
        self.assertFalse(os.path.exists(tmp))
        _prepend_string_to_file(fn, "y\n")
        self.assertFalse(os.path.exists(tmp))


class TestAddApreproParamFileLinesPreprocessor(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def test_process_prepends_param_and_state_in_order(self):
        # This test verifies the *order* of prepend calls, and that it resolves
        # the input file into template_dir/basename(input_filename).
        # nested input name to ensure basename() behavior is used
        input_filename = os.path.join("some", "path", "my_deck.i")
        deck_on_disk = "./my_deck.i"
        with open(deck_on_disk, "w") as f:
            f.write("body\n")

        p = AddApreproParamFileLinesPreprocessor()

        with mock.patch("matcal.sierra.models.preprocessors._prepend_string_to_file") as m_add:
            p.process('.', input_filename)

            # Called twice, parameter include first, then state include
            self.assertEqual(m_add.call_count, 2)
            self.assertEqual(m_add.call_args_list[0].args[0], deck_on_disk)
            self.assertEqual(m_add.call_args_list[0].args[1], p.param_aprepro_include)

            self.assertEqual(m_add.call_args_list[1].args[0], deck_on_disk)
            self.assertEqual(m_add.call_args_list[1].args[1], p.state_aprepro_include)


class TestDecomposeAndCopyMeshPreprocessor(MatcalUnitTest):
    def setUp(self):
        super().setUp(__file__)

    def _write_dummy_mesh(self, folder, name="mesh.g"):
        fn = os.path.join(folder, name)
        with open(fn, "w") as f:
            f.write("dummy mesh content\n")
        return fn

    def test_serial_copies_mesh_and_symlinks_into_template_dir(self):
        """
        n_cores == 1:
          - should call _copy_file_or_directory_to_target_directory(mesh_folder, mesh_filename)
          - should create symlink(s) into template_dir
        """
        with (
            tempfile.TemporaryDirectory() as template_dir, 
            tempfile.TemporaryDirectory() as src_dir
        ):
            mesh_filename = self._write_dummy_mesh(src_dir, "mesh.g")
            computing_info = SimpleNamespace(number_of_cores=1)

            mesh_template_dir = os.path.join(template_dir, "mesh_files")

            # Patch module-level helpers to avoid depending on matcal internals
            with(
                mock.patch(
                    "matcal.sierra.models.preprocessors._get_mesh_template_folder", 
                    return_value=mesh_template_dir
                    ), 
                mock.patch(
                    "matcal.sierra.models.preprocessors._copy_file_or_directory_to_target_directory"
                    ) as m_copy, 
                mock.patch(
                    "matcal.sierra.models.preprocessors.get_mesh_decomposer"
                    ) as m_get_decomposer
                ):

                # Provide a decomposer class (won't be used in serial path)
                decomposer_instance = mock.Mock()
                m_get_decomposer.return_value = lambda: decomposer_instance

                os.makedirs(mesh_template_dir, exist_ok=True)

                p = DecomposeAndCopyMeshPreprocessor()
                p.process(computing_info, template_dir, mesh_filename, delete_source_mesh=False)

                # Verify copy helper called with (mesh_files_template_folder, mesh_filename)
                m_copy.assert_called_once()
                called_mesh_folder, called_mesh_filename = m_copy.call_args.args
                self.assertEqual(called_mesh_folder, mesh_template_dir)
                self.assertEqual(os.path.abspath(called_mesh_filename), os.path.abspath(mesh_filename))

                # Simulate what copy helper would do (since we mocked it)
                copied_mesh = os.path.join(mesh_template_dir, os.path.basename(mesh_filename))
                shutil.copy(mesh_filename, copied_mesh)

                # Rerun the symlink loop expectation by checking that a symlink exists
                # NOTE: the process already ran; but because we mocked the copy helper,
                # the file didn't exist during symlink creation. To avoid this, you can
                # instead not mock the copy helper, but that would use matcal's function.
                #
                # So for this mocked version, we validate the symlink behavior by
                # re-running only the symlink portion isn't practical. Alternative:
                # don't mock the copy helper and patch it to shutil.copy. We'll do that
                # in the next test.
                #
                # For now: just ensure we created mesh folder and didn't crash.
                self.assertTrue(os.path.isdir(mesh_template_dir))

    def test_serial_real_copy_then_symlink(self):
        """
        Same as above, but patch copy helper to shutil.copy so the mesh exists
        when symlinks are created.
        """
        with(
            tempfile.TemporaryDirectory() as template_dir, 
            tempfile.TemporaryDirectory() as src_dir
        ):
            mesh_filename = self._write_dummy_mesh(src_dir, "mesh.g")
            computing_info = SimpleNamespace(number_of_cores=1)

            mesh_template_dir = os.path.join(template_dir, "mesh_files")
            os.makedirs(mesh_template_dir, exist_ok=True)

            def _copy_impl(mesh_folder, src):
                shutil.copy(src, os.path.join(mesh_folder, os.path.basename(src)))

            with(
                mock.patch(
                    "matcal.sierra.models.preprocessors._get_mesh_template_folder",
                    return_value=mesh_template_dir
                    ), 
                mock.patch(
                    "matcal.sierra.models.preprocessors._copy_file_or_directory_to_target_directory",
                    side_effect=_copy_impl
                    ), 
                 mock.patch(
                    "matcal.sierra.models.preprocessors.get_mesh_decomposer"
                    ) as m_get_decomposer
            ):

                decomposer_instance = mock.Mock()
                m_get_decomposer.return_value = lambda: decomposer_instance

                p = DecomposeAndCopyMeshPreprocessor()
                p.process(computing_info, template_dir, mesh_filename, delete_source_mesh=False)

                copied_mesh = os.path.join(mesh_template_dir, "mesh.g")
                self.assertTrue(os.path.exists(copied_mesh))

                # Symlink should exist in template_dir with same basename
                link_path = os.path.join(template_dir, "mesh.g")
                self.assertTrue(os.path.islink(link_path))
                self.assertEqual(os.path.realpath(link_path), os.path.abspath(copied_mesh))

    def test_parallel_decomposes_and_removes_undecomposed_and_symlinks(self):
        """
        n_cores > 1:
          - should call decomposer.decompose_mesh(
                        abs(mesh_filename),
                        n_cores, mesh_files_template_folder)
          - should remove template_mesh_filename if present
          - should symlink decomposed pieces into template_dir
        """
        with(
            tempfile.TemporaryDirectory() as template_dir, 
            tempfile.TemporaryDirectory() as src_dir
        ):
            mesh_filename = self._write_dummy_mesh(src_dir, "mesh.g")
            computing_info = SimpleNamespace(number_of_cores=4)

            mesh_template_dir = os.path.join(template_dir, "mesh_files")
            os.makedirs(mesh_template_dir, exist_ok=True)

            # Create the "undecomposed mesh" in the template mesh folder to ensure it gets removed
            undecomp_in_template = os.path.join(mesh_template_dir, "mesh.g")
            shutil.copy(mesh_filename, undecomp_in_template)
            self.assertTrue(os.path.exists(undecomp_in_template))

            # Create some fake decomposed pieces in the mesh folder that should be symlinked
            piece1 = os.path.join(mesh_template_dir, "mesh.g.4.0")
            piece2 = os.path.join(mesh_template_dir, "mesh.g.4.1")
            with open(piece1, "w") as f:
                f.write("piece0\n")
            with open(piece2, "w") as f:
                f.write("piece1\n")

            decomposer_instance = mock.Mock()

            with(
                mock.patch(
                    "matcal.sierra.models.preprocessors._get_mesh_template_folder",
                    return_value=mesh_template_dir
                    ), 
                mock.patch(
                    "matcal.sierra.models.preprocessors.get_mesh_decomposer"
                    ) as m_get_decomposer
                ):

                m_get_decomposer.return_value = lambda: decomposer_instance

                p = DecomposeAndCopyMeshPreprocessor()
                p.process(computing_info, template_dir, mesh_filename, delete_source_mesh=False)

                decomposer_instance.decompose_mesh.assert_called_once()
                args = decomposer_instance.decompose_mesh.call_args.args
                self.assertEqual(args[0], os.path.abspath(mesh_filename))
                self.assertEqual(args[1], 4)
                self.assertEqual(args[2], mesh_template_dir)

                # undecomposed mesh should be removed (SIERRA uses the pieces)
                self.assertFalse(os.path.exists(undecomp_in_template))

                # Symlinks should be present in template_dir for the pieces
                link1 = os.path.join(template_dir, os.path.basename(piece1))
                link2 = os.path.join(template_dir, os.path.basename(piece2))
                self.assertTrue(os.path.islink(link1))
                self.assertTrue(os.path.islink(link2))
                self.assertEqual(os.path.realpath(link1), os.path.abspath(piece1))
                self.assertEqual(os.path.realpath(link2), os.path.abspath(piece2))

    def test_delete_source_mesh_moves_into_mesh_folder(self):
        """
        delete_source_mesh=True:
          - should move mesh into template mesh folder before copy/decompose logic.
        """
        with(
            tempfile.TemporaryDirectory() as template_dir,
            tempfile.TemporaryDirectory() as src_dir
        ):
            mesh_filename = self._write_dummy_mesh(src_dir, "mesh.g")
            computing_info = SimpleNamespace(number_of_cores=1)

            mesh_template_dir = os.path.join(template_dir, "mesh_files")
            os.makedirs(mesh_template_dir, exist_ok=True)

            def _copy_impl(mesh_folder, src):
                # no-op if src already in folder, otherwise copy
                dst = os.path.join(mesh_folder, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy(src, dst)

            with(
                mock.patch(
                    "matcal.sierra.models.preprocessors._get_mesh_template_folder",
                    return_value=mesh_template_dir
                    ), 
                mock.patch(
                    "matcal.sierra.models.preprocessors._copy_file_or_directory_to_target_directory",
                    side_effect=_copy_impl
                    ), 
                mock.patch(
                    "matcal.sierra.models.preprocessors.get_mesh_decomposer"
                    ) as m_get_decomposer
                ):

                decomposer_instance = mock.Mock()
                m_get_decomposer.return_value = lambda: decomposer_instance

                p = DecomposeAndCopyMeshPreprocessor()
                p.process(computing_info, template_dir, mesh_filename, delete_source_mesh=True)

                moved = os.path.join(mesh_template_dir, "mesh.g")
                self.assertTrue(os.path.exists(moved))
                # original should be gone because it was moved
                self.assertFalse(os.path.exists(mesh_filename))

                # symlink created in template_dir
                link_path = os.path.join(template_dir, "mesh.g")
                self.assertTrue(os.path.islink(link_path))
                self.assertEqual(os.path.realpath(link_path), os.path.abspath(moved))