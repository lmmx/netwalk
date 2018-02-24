from .context import netwalk
import numpy as np
from netwalk.sym_dict import out_1_state, out_2h_state, out_4_state, \
h_state, on_state, terminal, l_wire, c_wire, t_wire, server

import unittest

class OutStateTestSuite(unittest.TestCase):
    """
    Test cases for the various out state classes.
    """

    def test_out_1_state_to_dir(self):
        """
        Test the ``out_to_direction`` static method of ``out_1_state``
        """
        self.assertEqual([out_1_state.out_to_direction(x) \
            for x in np.arange(4)], ['up', 'right', 'down', 'left'])
        with self.assertRaises(ValueError):
            out_1_state.out_to_direction(5)
        with self.assertRaises(ValueError):
            out_1_state.out_to_direction(-1)

    def test_out_1_state(self):
        """
        Test the initialisation of out_1_state.
        """
        self.assertEqual(out_1_state(0).out, 0)
        self.assertEqual(out_1_state(0).direction, 'up')
        self.assertTrue(np.array_equal(out_1_state(0).all_dirs,
            np.array([1,0,0,0], dtype='bool')))
        self.assertTrue(np.array_equal(out_1_state(1).all_dirs,
            np.array([0,1,0,0], dtype='bool')))
        self.assertTrue(np.array_equal(out_1_state(2).all_dirs,
            np.array([0,0,1,0], dtype='bool')))
        self.assertTrue(np.array_equal(out_1_state(3).all_dirs,
            np.array([0,0,0,1], dtype='bool')))

    def test_out_2h_state_to_dir(self):
        """
        Test the ``out_to_direction`` static method of ``out_2h_state``
        """
        self.assertEqual(out_2h_state.out_to_direction(True),
                         'left and right')
        self.assertEqual(out_2h_state.out_to_direction(False),
                         'up and down')
        self.assertTrue(np.array_equal(out_2h_state.out_to_all_dirs(True),
            np.array([0,1,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(out_2h_state.out_to_all_dirs(False),
            np.array([1,0,1,0], dtype='bool')))

    def test_out_2h_state(self):
        """
        Test the initialisation of out_2h_state.
        """
        self.assertTrue(out_2h_state(True).horizontal)
        self.assertFalse(out_2h_state(False).horizontal)
        self.assertEqual(out_2h_state(True).direction, 'left and right')
        self.assertEqual(out_2h_state(False).direction, 'up and down')
        self.assertTrue(np.array_equal(out_2h_state(True).all_dirs,
            np.array([0,1,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(out_2h_state(False).all_dirs,
            np.array([1,0,1,0], dtype='bool')))

    def test_out_4_state_to_dir(self):
        """
        Test the ``out_to_direction`` static method of ``out_4_state``
        on some cases (not going to enumerate all possibilities).
        """
        self.assertEqual(out_4_state.out_to_direction(
            np.array([0,0,0,0], dtype='bool')), [])
        self.assertEqual(out_4_state.out_to_direction(
            np.array([0,0,0,1], dtype='bool')), ['left'])
        self.assertEqual(out_4_state.out_to_direction(
            np.array([0,0,1,0], dtype='bool')), ['down'])
        self.assertEqual(out_4_state.out_to_direction(
            np.array([0,1,0,0], dtype='bool')), ['right'])
        self.assertEqual(out_4_state.out_to_direction(
            np.array([1,0,0,0], dtype='bool')), ['up'])
        self.assertEqual(len(out_4_state.out_to_direction(
            np.array([1,1,1,1], dtype='bool'))), 4)
        with self.assertRaises(TypeError):
            out_4_state.out_to_direction(
                np.array([2,2,2,2]))
        with self.assertRaises(ValueError):
            out_4_state.out_to_direction(
                np.array([1], dtype='bool'))
        with self.assertRaises(ValueError):
            out_4_state.out_to_direction(
                np.array([1,1], dtype='bool'))
        with self.assertRaises(ValueError):
            out_4_state.out_to_direction(
                np.array([1,1,1], dtype='bool'))
        with self.assertRaises(ValueError):
            out_4_state.out_to_direction(
                np.array([1,1,1,1,1], dtype='bool'))

    def test_out_4_state(self):
        """
        Test the initialisation of out_4_state.
        """
        self.assertTrue(np.array_equal(
            out_4_state(np.array([0,1,0,1], dtype='bool')).out,
            np.array([0,1,0,1], dtype='bool')))
        self.assertEqual(
            out_4_state(np.array([0,1,0,1], dtype='bool')).direction,
            ['right','left'])
        self.assertEqual(
            out_4_state(np.array([0,1,0,1], dtype='bool')).__repr__(),
            'right and left')

class hStateTestSuite(unittest.TestCase):
    """
    Test cases for the h_state wrapper class.
    """

    def test_h_state(self):
        """
        Test the instantiation of a ``h_state`` wrapper class.
        """
        self.assertEqual(h_state(True).__repr__(), 'horizontal')
        self.assertEqual(h_state(False).__repr__(), 'vertical')
        self.assertTrue(np.array_equal(h_state(True).horizontal.all_dirs,
            np.array([0,1,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(h_state(False).horizontal.all_dirs,
            np.array([1,0,1,0], dtype='bool')))
        self.assertEqual(h_state(True).horizontal.direction, 'left and right')
        self.assertEqual(h_state(False).horizontal.direction, 'up and down')
        self.assertTrue(h_state(True).horizontal.horizontal)
        self.assertFalse(h_state(False).horizontal.horizontal)

class OnStateTestSuite(unittest.TestCase):
    """
    Test cases for the ``on_state`` class.
    """

    def test_on_state(self):
        """
        Test the instantiation of the ``on_state`` class.
        """
        self.assertTrue(on_state(True).on)
        self.assertFalse(on_state(False).on)
        self.assertEqual(on_state(True).__repr__(), 'on')
        self.assertEqual(on_state(False).__repr__(), 'off')

class ComponentsTestSuite(unittest.TestCase):
    """
    Test cases for all component classes (wires, node, and server).
    """

    def test_server(self):
        """
        Test the instantiation of the ``server`` component class.
        """
        # ...

    def test_terminal(self):
        """
        Test the instantiation of the ``terminal`` component class.
        """
        self.assertEqual(terminal(0, True).out.out, 0)
        self.assertTrue(np.array_equal(terminal(0, True).directions,
            np.array([1,0,0,0], dtype='bool')))
        self.assertTrue(terminal(0, True).state.on)
        self.assertFalse(terminal(0, False).state.on)
        self.assertEqual(terminal(0, True).__repr__(),
            'A terminal pointing up, switched on.')
        self.assertEqual(terminal(1, False).__repr__(),
            'A terminal pointing right, switched off.')
        self.assertEqual(terminal(2, True).__repr__(),
            'A terminal pointing down, switched on.')
        self.assertEqual(terminal(3, False).__repr__(),
            'A terminal pointing left, switched off.')
        with self.assertRaises(ValueError):
            terminal(-1, True)
        with self.assertRaises(ValueError):
            terminal(4, True)

    def test_l_wire(self):
        """
        Test the instantiation of the ``l_wire`` component class.
        """
        # this state nesting is ugly but I guess it serves a purpose...
        self.assertTrue(l_wire(True).horizontal.horizontal.horizontal)
        self.assertFalse(l_wire(False).horizontal.horizontal.horizontal)
        self.assertTrue(np.array_equal(l_wire(True).directions,
            np.array([0,1,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(l_wire(False).directions,
            np.array([1,0,1,0], dtype='bool')))
        self.assertEqual(l_wire(True).__repr__(),
            'A line wire pointing horizontally.')
        self.assertEqual(l_wire(False).__repr__(),
            'A line wire pointing vertically.')

    def test_c_wire(self):
        """
        Test the instantiation of the ``c_wire`` component class.
        """
        self.assertTrue(np.array_equal(c_wire(0).corner,
            np.array([1,0,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(c_wire(1).corner,
            np.array([1,1,0,0], dtype='bool')))
        self.assertTrue(np.array_equal(c_wire(2).corner,
            np.array([0,1,1,0], dtype='bool')))
        self.assertTrue(np.array_equal(c_wire(3).corner,
            np.array([0,0,1,1], dtype='bool')))
        self.assertEqual(c_wire(0).out.__repr__(), 'up and left')
        self.assertEqual(c_wire(0).directions, ['up', 'left'])
        self.assertEqual(c_wire(1).out.__repr__(), 'up and right')
        self.assertEqual(c_wire(1).directions, ['up', 'right'])
        self.assertEqual(c_wire(2).out.__repr__(), 'right and down')
        self.assertEqual(c_wire(2).directions, ['right', 'down'])
        self.assertEqual(c_wire(3).out.__repr__(), 'down and left')
        self.assertEqual(c_wire(3).directions, ['down', 'left'])
        with self.assertRaises(ValueError):
            c_wire(-1)
        with self.assertRaises(ValueError):
            c_wire(4)

    def test_c_wire_corner_parser(self):
        """
        Test the ``parse_corner`` static method of the ``c_wire`` class.
        """
        self.assertTrue(np.array_equal(c_wire.parse_corner(0),
            np.array([1,0,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(c_wire.parse_corner(1),
            np.array([1,1,0,0], dtype='bool')))
        self.assertTrue(np.array_equal(c_wire.parse_corner(2),
            np.array([0,1,1,0], dtype='bool')))
        self.assertTrue(np.array_equal(c_wire.parse_corner(3),
            np.array([0,0,1,1], dtype='bool')))
        with self.assertRaises(ValueError):
            c_wire.parse_corner(-1)
        with self.assertRaises(ValueError):
            c_wire.parse_corner(4)


    def test_t_wire(self):
        """
        Test the instantiation of the ``t_wire`` component class.
        """
        self.assertTrue(np.array_equal(t_wire(0).facing,
            np.array([1,1,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(t_wire(1).facing,
            np.array([1,1,1,0], dtype='bool')))
        self.assertTrue(np.array_equal(t_wire(2).facing,
            np.array([0,1,1,1], dtype='bool')))
        self.assertTrue(np.array_equal(t_wire(3).facing,
            np.array([1,0,1,1], dtype='bool')))
        self.assertEqual(t_wire(0).out.__repr__(), 'up, right, and left')
        self.assertEqual(t_wire(0).directions, ['up', 'right', 'left'])
        self.assertEqual(t_wire(0).__repr__(),
            'A T-shaped wire pointing up, right, and left.')
        self.assertEqual(t_wire(1).out.__repr__(), 'up, right, and down')
        self.assertEqual(t_wire(1).directions, ['up', 'right', 'down'])
        self.assertEqual(t_wire(1).__repr__(),
            'A T-shaped wire pointing up, right, and down.')
        self.assertEqual(t_wire(2).out.__repr__(), 'right, down, and left')
        self.assertEqual(t_wire(2).directions, ['right', 'down', 'left'])
        self.assertEqual(t_wire(2).__repr__(),
            'A T-shaped wire pointing right, down, and left.')
        self.assertEqual(t_wire(3).out.__repr__(), 'up, down, and left')
        self.assertEqual(t_wire(3).directions, ['up', 'down', 'left'])
        self.assertEqual(t_wire(3).__repr__(),
            'A T-shaped wire pointing up, down, and left.')
        with self.assertRaises(ValueError):
            t_wire(-1)
        with self.assertRaises(ValueError):
            t_wire(4)

    def test_t_wire_facing_parser(self):
        """
        Test the ``parse_facing`` static method of the ``t_wire`` class.
        """
        self.assertTrue(np.array_equal(t_wire.parse_facing(0),
            np.array([1,1,0,1], dtype='bool')))
        self.assertTrue(np.array_equal(t_wire.parse_facing(1),
            np.array([1,1,1,0], dtype='bool')))
        self.assertTrue(np.array_equal(t_wire.parse_facing(2),
            np.array([0,1,1,1], dtype='bool')))
        self.assertTrue(np.array_equal(t_wire.parse_facing(3),
            np.array([1,0,1,1], dtype='bool')))
        with self.assertRaises(ValueError):
            t_wire.parse_facing(-1)
        with self.assertRaises(ValueError):
            t_wire.parse_facing(4)

##########################################################################

if __name__ == '__main__':
    unittest.main()
