from snowdrop.src.preprocessor.language import minilang, functions

class NumericEval:
    """This class defines several evaluation methods."""
    
    def __init__(self, d, minilang=minilang):

        self.d = d # dictionary of substitutions
        for k,v in d.items():
            assert(isinstance(k, str))
        for k,v in functions.items():
            d[k] = v

        self.minilang = minilang

    def __call__(self, s):
        """
        """
        return self.eval(s)

    def eval(self, struct):
        """Evaluate structure."""
        from snowdrop.src.preprocessor.language import types
        t = struct.__class__.__name__
        
        if t in types:
            return struct.eval(self.d)
            
#        tt = tuple(self.minilang)
#        if isinstance(struct, tt):
#            return struct.eval(self.d)

        method_name = 'eval_' + t.lower()
        try:
            fun = getattr(self, method_name)

        except Exception:
            raise Exception("Unknown type {}".format(method_name))

        return fun(struct)

    def eval_scalarfloat(self, s):
        """Evaluate scalar."""
        return float(s)

    def eval_float(self, s):
        """Evaluate float."""
        return s
    
    def eval_float64(self, s):
        """Evaluate float."""
        return s
    
    def eval_int(self, s):
        """Evaluate integer."""
        return s

    def eval_str(self, s):
        """Evaluate string."""
        # not safe
        return eval(s, self.d)

    def eval_list(self, l):
        """Evaluate list."""
        return [self.eval(e) for e in l]

    def eval_dict(self, d):
        """Evaluate dictionary."""
        return {k: self.eval(e) for k, e in d.items()}
    
    def eval_commentedmap(self,d):
        """Evaluate commented map."""
        return {k: self.eval(e) for k, e in d.items()}

    def eval_ordereddict(self, s):
        """Evaluate ordered dictionary."""
        from collections import OrderedDict
        res = OrderedDict()
        for k in s.keys():
            v = self.eval(s[k])
            res[k] = v

        return res

    def eval_commentedseq(self, s):
        """Evaluate comments."""
        return self.eval_list(s)

    def eval_ndarray(self, array_in):
        """Evaluate ndarray."""
        import numpy as np
        array_out = np.zeros_like(array_in, dtype=float)
        nd = np.ndim(array_in)
        if nd == 1:
            for i in range(array_in.shape[0]):
                array_out[i] = self.eval(array_in[i])
        else:
            for i in range(array_in.shape[0]):
                for j in range(array_in.shape[1]):
                    array_out[i,j] = self.eval(array_in[i,j])
        return array_out

    def eval_nonetype(self, none):
        return None


if __name__ == '__main__':
    """
    Main entry point
    """
    from collections import OrderedDict
    options = OrderedDict(
        smin= ['x',0.0],
        smax= ['y','x'],
        orders= [40,40],
        markov=dict(a=12.0, b=0.9)
    )
    d = {'x': 0.01, 'y': 10.0}
    print( NumericEval(d)(options) )
