import json
import logging
import string
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dash import html

from deepcave.utils.cache import Cache
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.data_structures import update_dict
from deepcave.utils.files import make_dirs
from deepcave.utils.hash import file_to_hash, string_to_hash
from deepcave.utils.layout import (
    get_checklist_options,
    get_radio_options,
    get_select_options,
    get_slider_marks,
)
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_color, hex_to_rgb
from deepcave.utils.util import get_random_string, matplotlib_to_html_image


class TestCache(unittest.TestCase):
    def test_cache_from_new_file(self):
        cache_file = Path("tests/cache_test/cache.json")
        cache_file.unlink(missing_ok=True)

        # Load with new file
        self.assertFalse(cache_file.exists())
        cache = Cache(cache_file)

        # Set values
        cache.set("a", "b", "c", value=4)

        # Check whether values were written to file
        self.assertTrue(cache_file.exists())
        with cache_file.open() as f:
            dict_from_file = json.load(f)

        self.assertIn("a", dict_from_file)
        self.assertIn("b", dict_from_file["a"])
        self.assertIn("c", dict_from_file["a"]["b"])
        self.assertEqual(4, dict_from_file["a"]["b"]["c"])

        # Cleanup
        cache_file.unlink()

    def test_cache_from_existing_file(self):
        # Prepare existing file
        cache_file = Path("tests/cache_test/cache2.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('{"d": {"e": {"f": 32}}}')

        # Load with exising file
        self.assertTrue(cache_file.exists())
        cache = Cache(cache_file)

        # Get values
        value = cache.get("d", "e", "f")
        self.assertEqual(32, value)

        # Cleanup
        cache_file.unlink()

    def test_cache_defaults(self):
        cache_file = Path("tests/cache_test/cache3.json")
        cache_file.unlink(missing_ok=True)

        # Load with new file
        self.assertFalse(cache_file.exists())
        defaults = {"i": {"j": 4}, "k": {"l": {"m": 42}}, "v": 9}
        cache = Cache(cache_file, defaults=defaults)

        # Test get values
        self.assertEqual(4, cache.get("i", "j"))
        self.assertEqual(42, cache.get("k", "l", "m"))
        self.assertEqual(9, cache.get("v"))

        # Check whether values were written to file
        self.assertTrue(cache_file.exists())
        with cache_file.open() as f:
            dict_from_file = json.load(f)

        self.assertIn("i", dict_from_file)
        self.assertIn("k", dict_from_file)
        self.assertIn("v", dict_from_file)
        self.assertIn("j", dict_from_file["i"])
        self.assertIn("l", dict_from_file["k"])
        self.assertIn("m", dict_from_file["k"]["l"])
        self.assertEqual(4, dict_from_file["i"]["j"])
        self.assertEqual(42, dict_from_file["k"]["l"]["m"])
        self.assertEqual(9, dict_from_file["v"])

        # Cleanup
        cache_file.unlink()

    def test_cache_has(self):
        cache_file = Path("tests/cache_test/cache4.json")
        cache_file.unlink(missing_ok=True)

        # Load with new file
        self.assertFalse(cache_file.exists())
        defaults = {"i": {"j": 4}, "k": {"l": {"m": 42}}, "v": 9}
        cache = Cache(cache_file, defaults=defaults)

        # Test has values
        self.assertTrue(cache.has("i"))
        self.assertTrue(cache.has("i", "j"))
        self.assertTrue(cache.has("k"))
        self.assertTrue(cache.has("k", "l"))
        self.assertTrue(cache.has("k", "l", "m"))
        self.assertTrue(cache.has("v"))

        # Cleanup
        cache_file.unlink()

    def test_cache_file_none(self):
        """Cache should still work, even when file is None"""
        cache = Cache(None)

        cache.set("1", "2", "3", value=4)
        self.assertEqual(4, cache.get("1", "2", "3"))


class TestCompression(unittest.TestCase):
    def test_list_conversion(self):
        a = [1, 2, 3, 4, 5, 6]
        a_ser = serialize(a)
        self.assertEqual("[1,2,3,4,5,6]", a_ser)
        a_cycled = deserialize(a_ser, dtype=list)
        self.assertIsInstance(a_cycled, list)
        for i in range(len(a)):
            self.assertEqual(a[i], a_cycled[i])

    def test_dataframe_conversion(self):
        x = [1, 2, None]
        y = ["a", "b", "c"]
        df = pd.DataFrame([x, y])

        df_ser = serialize(df)
        self.assertEqual(
            "{" '"0":{"0":1,"1":"a"},' '"1":{"0":2,"1":"b"},' '"2":{"0":null,"1":"c"}' "}",
            df_ser,
        )
        df_cycled = deserialize(df_ser, dtype=pd.DataFrame)
        self.assertIsInstance(df_cycled, pd.DataFrame)
        self.assertTrue(all((df_cycled.to_numpy() == df.to_numpy()).reshape(-1)))


class TestDataStructures(unittest.TestCase):
    def test_update_dict(self):
        a = {"a": {"b": 1, "c": 2, "d": 3}}
        b = {"a": {"b": 4, "e": 5}, "b": {"f": 6}}

        update_dict(a, b)
        self.assertIn("a", a)
        self.assertIn("b", a)
        a_a = a["a"]
        self.assertEqual(4, a_a["b"])
        self.assertEqual(2, a_a["c"])
        self.assertEqual(3, a_a["d"])
        self.assertEqual(5, a_a["e"])

        a_b = a["b"]
        self.assertEqual(6, a_b["f"])


class TestFiles(unittest.TestCase):
    def test_make_dirs(self):
        def _test_path_procedure(argument, is_file=False):
            path = Path(argument)
            if is_file:
                folder = path.parent
            else:
                folder = path
            # Make sure that folder does not exist
            if folder.exists():
                folder.rmdir()

            # Test
            self.assertFalse(folder.exists())
            make_dirs(argument)
            self.assertTrue(folder.exists())
            if is_file:
                self.assertFalse(path.exists())

            # Cleanup
            folder.rmdir()

        _test_path_procedure("tests/new_folder")
        _test_path_procedure("tests/new_folder/file.txt", is_file=True)


class TestHash(unittest.TestCase):
    def test_string_to_hash(self):
        a = string_to_hash("hello")
        self.assertIsInstance(a, str)

        b = string_to_hash("world")
        self.assertNotEqual(a, b)

        b = string_to_hash("Hello")
        self.assertNotEqual(a, b)

        b = string_to_hash("hello")
        self.assertEqual(a, b)

    def test_file_to_hash(self):
        file = Path(__file__)
        a = file_to_hash(file)
        self.assertGreater(len(a), 5)
        self.assertIsInstance(a, str)


class TestLayout(unittest.TestCase):
    @unittest.SkipTest
    def test_get_slider_marks(self):
        # TODO(dwoiwode): Currently does not work as expected?
        default_marks = get_slider_marks()
        self.assertIsInstance(default_marks, dict)
        a = default_marks[0]
        self.assertEqual("None", a)

        abcde = list("ABCDE")
        marks = get_slider_marks(abcde)
        self.assertEqual(5, len(marks))
        for i, c in enumerate(abcde):
            self.assertEqual(c, marks[i])

        alphabet = string.ascii_uppercase
        marks = get_slider_marks(list(alphabet), 10)
        self.assertEqual(10, len(marks))

    def _test_get_select_options(self, method):
        # Test empty
        options = method(labels=None, values=None, binary=False)
        self.assertIsInstance(options, list)
        self.assertEqual(0, len(options))

        # Test binary
        options = method(binary=True)
        self.assertIsInstance(options, list)
        self.assertEqual(2, len(options))
        no, yes = sorted(options, key=lambda o: o["value"])
        self.assertTrue(yes["value"])
        self.assertFalse(no["value"])

        # Test copy labels
        labels = list("ABCDEF")
        options = method(labels)
        self.assertEqual(len(labels), len(options))
        for expected, actual in zip(labels, options):
            self.assertIsInstance(actual, dict)
            self.assertEqual(expected, actual["label"])
            self.assertEqual(expected, actual["value"])

        # Test copy values
        values = list("12345678")
        options = method(values=values)
        self.assertEqual(len(values), len(options))
        for expected, actual in zip(values, options):
            self.assertIsInstance(actual, dict)
            self.assertEqual(expected, actual["label"])
            self.assertEqual(expected, actual["value"])

        # Test labels + values
        labels = list("ABCDEFGHIJ")
        values = list("1234567890")
        options = method(labels, values)
        self.assertEqual(len(labels), len(options))
        for expected_label, expected_value, actual in zip(labels, values, options):
            self.assertIsInstance(actual, dict)
            self.assertEqual(expected_label, actual["label"])
            self.assertEqual(expected_value, actual["value"])

        # Test unequal length labels + values
        labels = list("ABCDEFGHIJ")
        values = list("1234567")
        self.assertRaises(ValueError, lambda: method(labels, values))

    def test_get_select_options(self):
        self._test_get_select_options(get_select_options)

    def test_get_radio_options(self):
        self._test_get_select_options(get_radio_options)

    def test_get_checklist_options(self):
        self._test_get_select_options(get_checklist_options)


class TestLogger(unittest.TestCase):
    def test_get_logger(self):
        logger = get_logger("TestLogger")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual("TestLogger", logger.name)

    def test_logging_config(self):
        mpl_logger = get_logger("matplotlib")
        self.assertEqual(logging.INFO, mpl_logger.level)
        self.assertFalse(mpl_logger.propagate)

        plugin_logger = get_logger("src.plugins")
        self.assertEqual(logging.DEBUG, plugin_logger.level)
        self.assertFalse(plugin_logger.propagate)


class TestStyledPlottly(unittest.TestCase):
    def test_hex_to_rgb(self):
        def assert_color(hex_code, expected_r, expected_g, expected_b):
            r, g, b = hex_to_rgb(hex_code)
            self.assertEqual(
                expected_r,
                r,
                f"r value does not match for {hex_code} (wanted {expected_r}, got {r})",
            )
            self.assertEqual(
                expected_g,
                g,
                f"g value does not match for {hex_code} (wanted {expected_g}, got {g})",
            )
            self.assertEqual(
                expected_b,
                b,
                f"b value does not match for {hex_code} (wanted {expected_b}, got {b})",
            )

        assert_color("#000000", 0, 0, 0)
        assert_color("#FFFFFF", 255, 255, 255)
        assert_color("#ffffff", 255, 255, 255)
        assert_color("#123456", 18, 52, 86)

        self.assertRaises(ValueError, lambda: hex_to_rgb("#0g0000"))
        self.assertRaises(ValueError, lambda: hex_to_rgb("000000"))

    def test_get_color(self):
        color_str = get_color(0)
        self.assertEqual("rgba(99, 110, 250, 1)", color_str)

        color_str = get_color(1, 0.3)
        self.assertEqual("rgba(239, 85, 59, 0.3)", color_str)

        self.assertRaises(IndexError, lambda: get_color(500))


class TestUtil(unittest.TestCase):
    def test_random(self):
        # Test Length
        a = get_random_string(10)
        self.assertIsInstance(a, str)
        self.assertEqual(10, len(a))

        # Test random
        b = get_random_string(10)
        self.assertIsInstance(b, str)
        self.assertEqual(10, len(b))
        self.assertNotEqual(a, b)

        # Test different length
        c = get_random_string(132)
        self.assertIsInstance(c, str)
        self.assertEqual(132, len(c))

        # Test Exception
        self.assertRaises(ValueError, lambda: get_random_string(-1))

    def test_matplotlib_to_html(self):
        fig = plt.Figure()
        ax = fig.gca()
        x = [1, 2, 3, 4, 5]
        y = [xx**2 for xx in x]
        ax.plot(x, y)

        html_img = matplotlib_to_html_image(fig)
        self.assertIsInstance(html_img, html.Img)

    @unittest.SkipTest
    def test_encode_data(self):
        # TODO(dwoiwode): Test with more knowledge about data structure
        pass

    @unittest.SkipTest
    def test_encode_data_with_cs(self):
        # TODO(dwoiwode): Test with more knowledge about data structure
        pass
