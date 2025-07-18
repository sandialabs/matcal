from matcal.core.object_factory import (IdentifierBase, IdentifierByTestFunction, 
                                       BasicIdentifier)
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class TestIdentifierBase():
    def __init__():
        pass

    class CommonTests(MatcalUnitTest):
        
        def setUp(self):
            super().setUp(__file__)
            self.identifier=self.get_identifier()

        def test_register(self):
            goal = {"a":1, 123:'golf', 'toga':'saree'}
            for k, v in goal.items():
                self.identifier.register(k, v)
            
            keys = self.identifier.keys
            self.assertEqual(len(keys), len(goal.keys()))
            for g_key in goal.keys():
                self.assertIn(g_key, list(keys))


class TestBasicIdentifier(TestIdentifierBase.CommonTests):
    
    def get_identifier(self):
        return BasicIdentifier()

    def test_register_and_retrieve(self):
        goal = {"a":1, 123:'golf', 'toga':'saree'}
        for k, v in goal.items():
            self.identifier.register(k, v)
        for g_key, g_val in goal.items():
            self.assertEqual(g_val, self.identifier.identify(g_key))

    def test_raise_key_error_if_key_not_present(self):
        goal = {"a":1, 123:'golf', 'toga':'saree'}
        for k, v in goal.items():
            self.identifier.register(k, v)
        with self.assertRaises(KeyError):
            self.identifier.identify('not_a_key')


class TestIdentifierByTestFunction(TestIdentifierBase.CommonTests):
    
    def get_identifier(self):
        self.default = "a"
        identifier = IdentifierByTestFunction(self.default)
        return identifier

    def test_register_and_retrieve(self):
        def return_true():
            return True
        def return_false():
            return False

        goal = {return_true:1, return_false:0}
        for k, v in goal.items():
            self.identifier.register(k, v)
        self.assertEqual(1, self.identifier.identify())

    def test_get_default(self):
        self.assertEqual(self.default, self.identifier.identify())

        def return_false():
            return False
        
        def also_return_false():
            return False
        
        goal = {return_false:1, also_return_false:0}
        for k, v in goal.items():
            self.identifier.register(k, v)
        self.assertEqual(self.default, self.identifier.identify())