from collections import OrderedDict

from matcal.core.input_file_writer import (InputFileLine, InputFileBlock, 
    InputFileTable, _BaseTypedInputFileBlock)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class InputFileLineTest(MatcalUnitTest):
    test_file = 'InputFileLine_test_file.txt'

    def setUp(self):
        super().setUp(__file__)

    def test_init_type_errors(self):
        self.assert_error_type(ValueError, InputFileLine, "")
        self.assert_error_type(TypeError, InputFileLine, None)

    def test_init_multi_element(self):
        A = InputFileLine("1", "2", "3")
        self.assertTrue(A.name == "1")
        for i in range(3):
            self.assertAlmostEqual(int(A._my_values[i]), i + 1)

    def test_init_multi_element_name(self):
        B = InputFileLine('Football', "Basketball", "Tennis", name='goat')
        self.assertEqual(B.name, 'goat')
        self.assertEqual(B._my_values[0], 'Football')
        self.assertEqual(B._my_values[1], 'Basketball')
        self.assertEqual(B._my_values[2], 'Tennis')

    def test_initSingelElementNameConversion(self):
        C = InputFileLine("This Will be A Name")
        self.assertEqual(C._my_values[0], "This Will be A Name")
        self.assertEqual(C.name, "This Will be A Name")

    def test_set(self):
        A = InputFileLine("Number of Cats")
        A.set("many", 2)
        A.set("too")

        self.assertEqual(A._my_values[1], "too")
        self.assertEqual(A._my_values[2], "many")

        A.set("not too", 1)
        A.set_at_end("cats")
        self.assertEqual(A._my_values[1], "not too")
        self.assertEqual(A._my_values[3], "cats")

        self.assert_error_type(ValueError, A.set, "no neg", -3)

    def assert_write(self, IFL, goal_line, indent):
        f = open(self.test_file, 'w')
        IFL.write(f, indent)
        f.close()
        f = open(self.test_file, 'r')
        line = f.readline()
        f.close()
        self.assertEqual(line, goal_line)
        self.remove_file(self.test_file)

    def test_writeSimple(self):
        A = InputFileLine("Interactions", 2)
        self.assert_write(A, "Interactions = 2\n", 0)

    def test_writeNoSymbol(self):
        B = InputFileLine("golf")
        B.set('putter')
        B.set('Driver', 5)
        B.suppress_symbol()
        self.assert_write(B, "golf putter Driver\n", 0)

    def test_writeDifferentSymbol(self):
        C = InputFileLine("Where", "can", "it", "be?")
        C.set_symbol("Its")
        C.set_symbol_location(4)
        C.set_at_end("in the car.")
        self.assert_write(C, "            Where can it be? Its in the car.\n", 3)

    def test_set_symbol_bad_type(self):
        C = InputFileLine("Where", "can", "it", "be?")
        with self.assertRaises(TypeError):
            C.set_symbol(1)

    def test_set_symbol_location_bad_type(self):
        C = InputFileLine("Where", "can", "it", "be?")
        with self.assertRaises(TypeError):
            C.set_symbol_location("a")

    def test_get_values(self):
        C = InputFileLine("Where", "can", "it", "be?")
        vals = C.get_values()
        self.assertEqual(vals, ["Where", "can", "it", "be?"])
        C = InputFileLine("Where", 1)
        vals = C.get_values()
        self.assertEqual(vals, ["Where", 1 ])


class InputFileBlockTest(MatcalUnitTest):
    test_file = 'InputFileBlock_test_file.txt'
    temp_goal_file = "InputFileBlock.goal"

    def setUp(self):
        super().setUp(__file__)

    def assert_write(self, IFB, goal):
        f = open(self.test_file, 'w')
        IFB.write(f)
        f.close()
        g = open(self.temp_goal_file, 'w')
        g.write(goal)
        g.close()

        self.assert_same_string_file(self.temp_goal_file, self.test_file)
        self.remove_file(self.temp_goal_file)
        self.remove_file(self.test_file)

    def test_init_set_name_all_entries(self):
        A = InputFileBlock("My Title", "my_name", True)
        self.assertEqual(A.name, "my_name")

    def test_get_title_string_print_title_and_name(self):
        A = InputFileBlock("My Title", "my_name")
        A.set_print_name(True)
        A.set_print_title(True)
        self.assertEqual(A._get_block_title_string(), "My Title my_name")
        A.set_print_title(False)
        self.assertEqual(A._get_block_title_string(), "my_name")
        A.set_print_title(True)
        A.set_print_name(False)
        self.assertEqual(A._get_block_title_string(), "My Title")

    def test_init_set_name(self):
        B = InputFileBlock("My Title")
        self.assertEqual(B.name, "My Title")

    def test_initBad(self):
        self.assert_error_type(ValueError, InputFileBlock, "")
        self.assert_error_type(TypeError, InputFileBlock, "stuff", 3)
        self.assert_error_type(TypeError, InputFileBlock, 12, "stuff")
        self.assert_error_type(TypeError, InputFileBlock, "as", 'asd', 1)

    def test_add_line_multiple(self):
        A = InputFileBlock("Variables")
        A.add_line(InputFileLine("alpha", 3))
        A.add_line(InputFileLine("beta", 3))
        with self.assertRaises(KeyError):
            A.add_line(InputFileLine("beta", 4))
            
        self.assertEqual(A.name, "Variables")
        self.assertEqual(A.get_line('alpha').name, 'alpha')
        self.assertEqual(A.get_line('beta').get_values()[-1], 3)
        self.assertEqual(A.get_line('beta').name, 'beta')

        A.add_line(InputFileLine("beta", 4), replace=True)
        self.assertEqual(A.get_line('beta').get_values()[-1], 4)

    def test_add_lines(self):
        A = InputFileBlock("Variables")
        lines_bad = [InputFileLine("alpha", 3), 
                     InputFileLine("beta", 3), 
                     InputFileLine("beta", 4)]
        with self.assertRaises(KeyError):
            A.add_lines(*lines_bad)

        lines = lines_bad[:-1]
        A = InputFileBlock("Variables")

        A.add_lines(*lines)

        self.assertEqual(A.name, "Variables")
        self.assertEqual(A.get_line('alpha').name, 'alpha')
        self.assertEqual(A.get_line('beta').get_values()[-1], 3)
        self.assertEqual(A.get_line('beta').name, 'beta')

        A.add_lines(*lines_bad, replace=True)
        self.assertEqual(A.get_line('beta').get_values()[-1], 4)

    def test_lines_property(self):
        A = InputFileBlock("Variables")
        l1 = InputFileLine("alpha", 3)
        l2 = InputFileLine("beta", 3)
        A.add_line(l1)
        A.add_line(l2)
        self.assertEqual(A.lines, {"alpha":l1, "beta":l2})

    def test_addLinesBad(self):
        B = InputFileBlock("Cats", 'animals')
        self.assertEqual(B.name, 'animals')
        self.assert_error_type(TypeError, B.add_line, "test")
        self.assert_error_type(TypeError, B.add_line, None)
        self.assert_error_type(TypeError, B.add_line, 1231234)

    def test_get_line_bad(self):
        A = InputFileBlock("Variables")
        A.add_line(InputFileLine("alpha", 3))
        A.add_line(InputFileLine("beta", 3))
        with self.assertRaises(KeyError):
            A.get_line("not a field")

    def test_get_line_value_bad_index(self):
        A = InputFileBlock("Variables")
        A.add_line(InputFileLine("alpha", 3))
        A.add_line(InputFileLine("beta", 3))
        with self.assertRaises(ValueError):
            A.get_line_value("alpha", 5)

    def test_set_symbol_to_string(self):
        A = InputFileBlock("Variables")
        A.add_line(InputFileLine("alpha", 3))
        A.add_line(InputFileLine("beta", 3))
        A.set_symbol_for_lines("!!")
        A_lines = A.get_string().split("\n")
        self.assertTrue("!!" in A_lines[1])
        self.assertTrue("!!" in A_lines[2])

    def test_set_symbol_to_None_to_suppress(self):
        A = InputFileBlock("Variables")
        line_1 = InputFileLine("alpha", 3)
        line_1.set_symbol("!!")
        A.add_line(line_1)
        line_2 = InputFileLine("beta", 3)
        line_2.set_symbol("!!")
        A.add_line(line_2)

        A.set_symbol_for_lines(None)
        A_lines = A.get_string().split("\n")
        self.assertFalse("!!" in A_lines[1])
        self.assertFalse("!!" in A_lines[2])

    def test_set_symbol_bad_value(self):
        A = InputFileBlock("Variables")
        line_1 = InputFileLine("alpha", 3)
        line_1.set_symbol("!!")
        A.add_line(line_1)
        line_2 = InputFileLine("beta", 3)
        line_2.set_symbol("!!")
        A.add_line(line_2)
        with self.assertRaises(TypeError):
            A.set_symbol_for_lines(1.0)
        
    def test_addTable(self):
        A = InputFileBlock("Variables")
        A.add_table(InputFileTable("TABLE", 3))
        A.add_table(InputFileTable("CHAIR", 1))
        self.assertIsInstance(A.get_table('TABLE'), InputFileTable)
        self.assertAlmostEqual(A.get_table('TABLE').get_num_col(), 3)
        self.assertIsInstance(A.get_table('CHAIR'), InputFileTable)
        self.assertAlmostEqual(A.get_table('CHAIR').get_num_col(), 1)

    def test_tables_property(self):
        A = InputFileBlock("Variables")
        tab1 = InputFileTable("TABLE", 3)
        tab2 = InputFileTable("CHAIR", 1)
        A.add_table(tab1)
        A.add_table(tab2)
        goal = {"TABLE":tab1, "CHAIR":tab2}
        for key in goal:
            self.assertEqual(A.tables[key], goal[key])

    def test_addTableBad(self):
        A = InputFileBlock("Variables")
        self.assert_error_type(TypeError, A.add_table, 12345)
        self.assert_error_type(TypeError, A.add_table, "")
        self.assert_error_type(TypeError, A.add_table, None)
        self.assert_error_type(TypeError, A.add_table, InputFileTable("TABLE", 3), 1231)
        self.assert_error_type(TypeError, A.add_table, InputFileTable("TABLE", 3), InputFileTable("TABLE2", 1))

    def test_subblocks_flat(self):
        A = InputFileBlock("Container")
        B = InputFileBlock("ItemA")
        C = InputFileBlock("ItemB", 'quail')

        A.add_subblock(B)
        A.add_subblock(C)

        self.assertEqual(A.get_subblock("ItemA").name, "ItemA")
        self.assertEqual(A.get_subblock('quail')._indent, 1)
        self.assertEqual(A.get_subblock('quail').name, "quail")

    def test_subblocks_add_repeat(self):
        A = InputFileBlock("Container")
        B = InputFileBlock("ItemB", "part")
        C = InputFileBlock("ItemC", 'part')

        A.add_subblock(B)
        with self.assertRaises(ValueError):
            A.add_subblock(C)
        
        A.add_subblock(C, replace=True)

        self.assertEqual(A.get_subblock("part").title, "ItemC")
        
    def test_subblocks_property(self):
        A = InputFileBlock("Container")
        B = InputFileBlock("ItemA")
        C = InputFileBlock("ItemB", 'quail')

        A.add_subblock(B)
        A.add_subblock(C)

        goal = {"ItemA":B, "quail":C}
        for key in goal:
            self.assertEqual(A.subblocks[key], goal[key])
        
    def test_remove_subblock(self):
        A = InputFileBlock("Container")
        B = InputFileBlock("ItemA")
        C = InputFileBlock("ItemB", 'quail')

        A.add_subblock(B)
        A.add_subblock(C)

        A.remove_subblock(B)
        with self.assertRaises(KeyError):
            A.get_subblock("ItemA")

        A.remove_subblock("quail")
        with self.assertRaises(KeyError):
            A.get_subblock("quail")

    def test_subblocks_bad(self):
        A = InputFileBlock("Container")
        with self.assertRaises(KeyError):
            A.get_line('not here')
        with self.assertRaises(KeyError):
            A.get_table('not here')
        with self.assertRaises(KeyError):
            A.get_subblock('not here')

        self.assert_error_type(ValueError, A.add_subblock, A)

        self.assert_error_type(TypeError, A.add_subblock, "123")
        block_1 = InputFileBlock("same subblock name")
        block_2 = InputFileBlock("same subblock name")
        A.add_subblock(block_1)
        self.assert_error_type(ValueError, A.add_subblock, block_2)

    def test_remove_nonexistent_subblock(self):
        A = InputFileBlock("Container")
        with self.assertRaises(KeyError):
            A.remove_subblock('not here')

    def test_subblocks_tall(self):
        D = InputFileBlock("SubItem1")
        E = InputFileBlock("SubItem2")
        F = InputFileBlock("SubItem3")

        E.add_subblock(F)
        D.add_subblock(E)
        self.assertEqual(D.get_subblock("SubItem2").get_subblock("SubItem3")._indent, 2)
        self.assertEqual(D.get_subblock("SubItem2")._indent, 1)
        self.assertEqual(D._indent, 0)

    def test_set_begin_end_parent(self):
        D = InputFileBlock("SubItem1", begin_end=True)
        E = InputFileBlock("SubItem2")
        F = InputFileBlock("SubItem3")

        E.add_subblock(F)
        D.add_subblock(E, set_begin_end_to_parent=True)
        self.assertTrue(D.get_subblock("SubItem2").get_subblock("SubItem3")._begin_end)
        self.assertTrue(D.get_subblock("SubItem2")._begin_end)
        
    def test_get_string_write_with_title(self):
        A = InputFileBlock("Solver")
        B = InputFileBlock("Type")
        C = InputFileBlock("Time")

        A.add_line(InputFileLine("tolerance", 1.0E-6))
        B.add_line(InputFileLine("CG"))
        B.get_line('CG').set_at_end("bicgstab")
        
        A.add_subblock(B)
        A.add_subblock(C)

        goal = "Solver\n"
        goal += "    tolerance = 1e-06\n\n"
        goal += "    Type\n" 
        goal += "        CG = bicgstab\n\n\n"
        goal += "    Time\n\n"

        self.assertEqual(A.get_string(), goal)
        self.assert_write(A, goal)

        A._begin_end = True
        B._begin_end = True
        C._begin_end = True

        goal = "Begin Solver\n"
        goal += "    tolerance = 1e-06\n\n"
        goal += "    Begin Type\n"
        goal += "        CG = bicgstab\n"  
        goal += "    End Type\n\n\n"
        goal += "    Begin Time\n"
        goal += "    End Time\n\n"
        goal += "End Solver\n"

        self.assertEqual(A.get_string(), goal)

    def test_write_name_instead_of_title(self):
        A = InputFileBlock("Solver", "my_solver_name")
        B = InputFileBlock("Type")
        C = InputFileBlock("Time")

        A.add_line(InputFileLine("tolerance", 1.0E-6))
        B.add_line(InputFileLine("CG"))
        B.get_line('CG').set_at_end("bicgstab")
        
        A.add_subblock(B)
        A.add_subblock(C)
        A.set_print_name()
        A.set_print_title(False)
        block_str = A.get_string()
        self.assertTrue("my_solver_name" in block_str)
        self.assertTrue("Solver" not in block_str)

    def test_set_name(self):
        block = InputFileBlock("title")
        self.assertEqual(block.name, "title")
        block.set_name("my_name")
        self.assertEqual(block.name, "my_name")
        
    def test_get_string_no_title(self):
        A = InputFileBlock("Solver", begin_end=True)
        B = InputFileBlock("Type", begin_end=False)
        C = InputFileBlock("Time", begin_end=True)

        A.add_line(InputFileLine("tolerance", 1.0E-6))
        B.add_line(InputFileLine("CG"))
        B.get_line('CG').set_at_end("bicgstab")

        A.add_subblock(B)
        A.add_subblock(C)
        self.assertTrue(B.print_title())
        B.set_print_title(False)
        self.assertFalse(B.print_title())

        goal = "Begin Solver\n"
        goal += "    tolerance = 1e-06\n\n"
        goal += "        CG = bicgstab\n\n\n"
        goal += "    Begin Time\n"
        goal += "    End Time\n\n"
        goal += "End Solver\n"

        self.assertEqual(A.get_string(), goal)

    def test_get_string_with_table(self):
        A = InputFileBlock("Solver")
        B = InputFileBlock("Type")
        C = InputFileBlock("Time")
        D = InputFileTable("table =", 3, begin_end_values=False)
        D.append_values(1,2,3)
        D.append_values(4,5,6)
        D.set_print_label()
        A.add_table(D)
        
        A.add_line(InputFileLine("tolerance", 1.0E-6))
        B.add_line(InputFileLine("CG"))
        B.get_line('CG').set_at_end("bicgstab")

        A.add_subblock(B)
        A.add_subblock(C)
        A._begin_end = True
        B._begin_end = True
        C._begin_end = True
        
        goal = "Begin Solver\n"
        goal += "    tolerance = 1e-06\n"
        goal += "    table =\n"
        goal += "        1 2 3\n"
        goal += "        4 5 6\n\n"
        goal += "    Begin Type\n"
        goal += "        CG = bicgstab\n"
        goal += "    End Type\n\n\n"
        goal += "    Begin Time\n"
        goal += "    End Time\n\n"
        goal += "End Solver\n"

        self.assertEqual(A.get_string(), goal)

    def test_add_lines_from_dict(self):
        test_dict = {"tree":True, "bush":False, 
                "leaves":100, "branches":3, 
                "leaves_per_branch":(10,20,70)}
        block = InputFileBlock("test")
        block.add_lines_from_dictionary(test_dict)

        self.assertTrue("tree" in block.lines)
        self.assertFalse("bush" in block.lines)
        self.assertEqual(block.get_line_value("leaves"), test_dict["leaves"])
        self.assertEqual(block.get_line_value("branches"), test_dict["branches"])
        self.assertEqual(block.get_line_value("leaves_per_branch"), 
                         test_dict["leaves_per_branch"][0])
        self.assertEqual(block.get_line_value("leaves_per_branch", 2), 
                         test_dict["leaves_per_branch"][1])
        self.assertEqual(block.get_line_value("leaves_per_branch", 3), 
                         test_dict["leaves_per_branch"][2])
        self.assertEqual(block.get_line("leaves_per_branch").get_values(), 
                         ["leaves_per_branch", 10, 20, 70])

    def test_write_input_to_file(self):
        A = InputFileBlock("Solver", begin_end=True)
        B = InputFileBlock("Type", begin_end=False)
        C = InputFileBlock("Time", begin_end=True)

        A.add_line(InputFileLine("tolerance", 1.0E-6))
        B.add_line(InputFileLine("CG"))
        B.get_line('CG').set_at_end("bicgstab")

        A.add_subblock(B)
        A.add_subblock(C)
        self.assertTrue(B.print_title())
        B.set_print_title(False)
        self.assertFalse(B.print_title())

        goal = "Begin Solver\n"
        goal += "    tolerance = 1e-06\n\n"
        goal += "        CG = bicgstab\n\n\n"
        goal += "    Begin Time\n"
        goal += "    End Time\n\n"
        goal += "End Solver\n"

        A.write_input_to_file("test.txt")

        self.assert_file_equals_string(goal, "test.txt")

class TypedInputFileBlockTest(MatcalUnitTest):

    class TestInputBlock(_BaseTypedInputFileBlock):
        type = 'test'
        required_keys = ["required"]
        default_values = {"default1":1 , "defaultstr":"str"}

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        block = self.TestInputBlock()
        self.assertEqual(block.name, "test")
        self.assertEqual(block._title, "test")
        self.assertEqual(block.get_line_value("default1"), 1)
        self.assertEqual(block.get_line_value("defaultstr"), "str")

    def test_missing_required(self):
        block = self.TestInputBlock()
        with self.assertRaises(ValueError):
            block.get_input_string()
        block.add_line(InputFileLine("required", True))
        input_str = block.get_input_string()        
        self.assertTrue(len(input_str) > 0)

    def test_get_subblock_by_type(self):
        block = self.TestInputBlock()
        block1 = self.TestInputBlock(name="test2")
        block2 = self.TestInputBlock(name="test3")
        block3 = self.TestInputBlock(name="test4")
        block.add_subblock(block1)
        block.add_subblock(block2)
        block.add_subblock(block3)

        block_test = block.get_subblock_by_type("test")
        self.assertEqual(block1, block_test)

    def test_remove_subblocks_by_type(self):
        block = self.TestInputBlock()
        block1 = self.TestInputBlock(name="test2")
        block2 = self.TestInputBlock(name="test3")
        block3 = self.TestInputBlock(name="test4")
        block.add_subblock(block1)
        block.add_subblock(block2)
        block.add_subblock(block3)

        block_test = block.remove_subblocks_by_type("test")
        self.assertEqual(block.subblocks, OrderedDict())        


class InputFileTableTest(MatcalUnitTest):
    test_file = "InputFileTable.test"
    goal_file = "InputFileTable.goal"

    def setUp(self):
        super().setUp(__file__)

    def test_init(self):
        A = InputFileTable("MyLabel", 2, name="my_name")
        self.assertEqual(A.name, "my_name")
        A = InputFileTable("MyLabel", 2)
        self.assertEqual(A.name, "MyLabel")

    def test_initBad(self):
        self.assert_error_type(InputFileTable.InvalidColumnNumberError, InputFileTable, "Label", "2")
        self.assert_error_type(InputFileTable.NonstringLabelError, InputFileTable, 2, "B")
        self.assert_error_type(InputFileTable.InvalidNameError, InputFileTable, "PP", 3, 4)
        self.assert_error_type(InputFileTable.NonstringLabelError, InputFileTable,"", 1, "asdf")
        self.assert_error_type(InputFileTable.InvalidColumnNumberError, InputFileTable, "Label", 0)
        self.assert_error_type(InputFileTable.InvalidColumnNumberError, InputFileTable, "Label", -1)

    def test_checkTableVales(self):
        A = InputFileTable("Label", 1)
        self.assertAlmostEqual(A.get_num_col(), 1)

        for i in range(4):
            A.append_values(i)

        for i in range(4):
            self.assertAlmostEqual(A._values[0][i], i)

    def test_checkTableMultiValue(self):
        A = InputFileTable("Label", 3)
        self.assertAlmostEqual(A.get_num_col(), 3)
        for i in range(6):
            A.append_values(i, 2 * i, i - 5)

        for i in range(6):
            self.assertAlmostEqual(A._values[0][i], i)
            self.assertAlmostEqual(A._values[1][i], 2 * i)
            self.assertAlmostEqual(A._values[2][i], i - 5)

    def test_wrongSize(self):
        A = InputFileTable("Label", 3)
        error_type = InputFileTable.ColumnNumberMismatchError
        self.assert_error_type(error_type, A.append_values, 1)
        self.assert_error_type(error_type, A.append_values, 1, 2, 3, 4, 5)
        self.assert_error_type(error_type, A.append_values)

    def test_checkTableWholeLists(self):
        A = InputFileTable("Label", 2)
        A.append_values(0, 0)
        L = [1, 2, 3, 4, 5]
        R = [11, 12, 13, 14, 15]
        A.append_lists(L, R)
        LS = [0, 1, 2, 3, 4, 5]
        RS = [0, 11, 12, 13, 14, 15]
        for i in range(6):
            self.assertAlmostEqual(A._values[0][i], LS[i])
            self.assertAlmostEqual(A._values[1][i], RS[i])

    def test_set_values(self):
        A = InputFileTable("Label", 2)
        L = [0, 1, 2, 3, 4, 5]
        R = [0, 11, 12, 13, 14, 15]
        A.set_values(L,R)
        LS = [0, 1, 2, 3, 4, 5]
        RS = [0, 11, 12, 13, 14, 15]
        for i in range(6):
            self.assertAlmostEqual(A._values[0][i], LS[i])
            self.assertAlmostEqual(A._values[1][i], RS[i])
        with self.assertRaises(A.ColumnNumberMismatchError):
            A.set_values(L,R,L)
        L.append(1)
        with self.assertRaises(A.ColumnLengthMismatchError):
            A.set_values(L,R)

    def test_badWholeLists(self):
        A = InputFileTable("Label", 2)
        L = [1, 2, 3, 4, 5]
        self.assert_error_type(A.ColumnNumberMismatchError, A.append_lists, L, L, L)
        self.assert_error_type(A.ColumnNumberMismatchError, A.append_lists)
        self.assert_error_type(A.ColumnNumberMismatchError, A.append_lists, L)
        self.assert_error_type(A.ColumnLengthMismatchError, A.append_lists, L, [1])

    def assert_write(self, IF, goal):
        f = open(self.test_file, 'w')
        IF.write(f, 1)
        f.close()
        g = open(self.goal_file, 'w')
        g.write(goal)
        g.close()

        self.assert_same_string_file(self.goal_file, self.test_file)
        self.remove_file(self.goal_file)
        self.remove_file(self.test_file)

    def test_writeTable(self):
        A = InputFileTable("Label", 2)
        L = [1, 2, 3, 4, 5]
        R = [11, 12, 13, 14, 15]
        A.append_lists(L, R)

        goal = "    Begin Values\n"
        goal += "        1 11\n"
        goal += "        2 12\n"
        goal += "        3 13\n" 
        goal += "        4 14\n"
        goal += "        5 15\n"
        goal += "    End Values\n"

        self.assert_write(A, goal)
