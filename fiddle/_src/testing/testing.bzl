"""Macros to make fiddle configs trivial to maintain through testing."""

load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

def fiddle_autotest(
        name,
        module):
    """Tests `module`'s configs to ensure they keep building."""

    program = ("'from fiddle.testing import autotest; " +
               "load_tests = autotest.load_tests; " +
               "autotest.main()'")

    native.genrule(
        name = name + "_genrule",
        srcs = [],
        outs = [name + ".py"],
        cmd = "echo " + program + " > $(location " + name + ".py)",
    )

    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = [
            module,
            "//third_party/py/fiddle/testing:autotest",
        ],
        args = ["--fiddle_config_module=$(location " + module + ")"],
    )
