from sympy.utilities.autowrap import *
from sympy.utilities.autowrap import _get_code_wrapper_class

def autowrap_and_store(expr, language='F95', backend='f2py', tempdir=None, args=None, flags=[],
		       verbose=False, helpers=[], # BEGIN MOD
		       mod_name=None, cleanup=False): # END MOD
    """
    Modified sympy.utilities.autowrap.autowrap()
    """
    import os
    autowrap_and_store.__doc__ += autowrap.__doc__

    code_generator = get_code_generator(language, "autowrap")
    CodeWrapperClass = _get_code_wrapper_class(backend)
    # BEGIN MOD
    # BEGIN COPY OF CodeWrapper.wrap_code
    def wrap_code(self, routine, helpers=[]):
	workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
	    #BEGIN SUBMOD
	    if not os.path.exists(self.filename+'.'+\
				  self.generator.code_extension):
		if verbose: print "Writing source code to: "+self.filename+'.'+\
				  self.generator.code_extension
		self._generate_code(routine, helpers)
		self._prepare_files(routine)
	    else:
		if verbose: print "Using previously created source file: "+\
		      self.filename+'.'+self.generator.code_extension
            # self._generate_code(routine, helpers)
            # self._prepare_files(routine)
	    #END SUBMOD
	    if verbose: print "Compiling binary..."
            self._process_files(routine)
	    if verbose: print "Compilation completed."
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter +=1
            os.chdir(oldwork)
            if not self.filepath:
                shutil.rmtree(workdir)

        return self._get_wrapped_function(mod)
    # END COPY OF CodeWrapper.wrapcode
    CodeWrapperClass.wrap_code = wrap_code
    CodeWrapperClass.module_name = mod_name
    CodeWrapperClass.filename = mod_name
    # END MOD
    code_wrapper = CodeWrapperClass(code_generator, tempdir, flags, verbose)
    code_wrapper.filename = tempdir #??
    try:
        routine  = Routine('autofunc', expr, args)
    except CodeGenArgumentListError, e:
        # if all missing arguments are for pure output, we simply attach them
        # at the end and try again, because the wrappers will silently convert
        # them to return values anyway.
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, OutputArgument):
                raise
            new_args.append(missing.name)
        routine  = Routine('autofunc', expr, args + new_args)

    helps = []
    for name, expr, args in helpers:
        helps.append(Routine(name, expr, args))

    #BEGIN MOD
    #return code_wrapper.wrap_code(routine, helpers=helps)

    retval = code_wrapper.wrap_code(routine, helpers=helps)
    if cleanup: os.unlink(os.path.join(tempdir,code_wrapper.filename+'.f90'))
    if cleanup: os.unlink(os.path.join(tempdir,code_wrapper.filename+'.h'))
    return retval
    #END MOD
