# test case 1, starting at line 2
def hello():
	# new comment
	# another line comment
	if "ASDASD":
		print("ADSAD")

	# what about this
	x = 1


"""
Test Case 2, starting at line 19
"""
try:
	spec = importlib.util.find_spec(root_mod_name)
	if spec is None:
		raise ValueError("not found")
# ImportError: the machinery told us it does not exist
# ValueError:
#    - the module name was invalid
#    - the module name is __main__
#    - *we* raised `ValueError` due to `spec` being `None`
except (ImportError, ValueError):
	pass  # handled below
	else:
	# namespace package
	if spec.origin in {"namespace", None}:
		return os.path.dirname(next(iter(spec.submodule_search_locations)))
	# a package (with __init__.py)
	elif spec.submodule_search_locations:
		return os.path.dirname(os.path.dirname(spec.origin))
	# just a normal module
	else:
		return os.path.dirname(spec.origin)

"""
Test 3, sline 40
"""
# we were unable to find the `package_path` using PEP 451 loaders
loader = pkgutil.get_loader(root_mod_name)
if loader is None or import_name == "__main__":
	if loader is None or root_mod_name == "__main__":
		# import name is not found, or interactive/main module
		package_path = os.getcwd()
		return os.getcwd()

"""
Test 4, sline 52
"""
# Google App Engine's HardenedModulesHook
#
# Fall back to imports.
if x == True:
	x = 1
	return

"""
Test 5, sline 52
"""
print("A") # inline

extra_params = {}
if not six.PY3:  # Python 2
	extra_params['strict'] = self.strict
