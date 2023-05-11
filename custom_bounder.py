from utils import grayToDecimal, grayCode
import copy
from typing import List
from math import floor

class Bounder(object):
    """Defines a basic bounding function for numeric lists.
    
    This callable class acts as a function that bounds a 
    numeric list between the lower and upper bounds specified.
    These bounds can be single values or lists of values. For
    instance, if the candidate is composed of five values, each
    of which should be bounded between 0 and 1, you can say
    ``Bounder([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])`` or just
    ``Bounder(0, 1)``. If either the ``lower_bound`` or 
    ``upper_bound`` argument is ``None``, the Bounder leaves 
    the candidate unchanged (which is the default behavior).
    
    As an example, if the bounder above were used on the candidate 
    ``[0.2, -0.1, 0.76, 1.3, 0.4]``, the resulting bounded
    candidate would be ``[0.2, 0, 0.76, 1, 0.4]``.
    
    A bounding function is necessary to ensure that all 
    evolutionary operators respect the legal bounds for 
    candidates. If the user is using only custom operators
    (which would be aware of the problem constraints), then 
    those can obviously be tailored to enforce the bounds
    on the candidates themselves. But the built-in operators
    make only minimal assumptions about the candidate solutions.
    Therefore, they must rely on an external bounding function
    that can be user-specified (so as to contain problem-specific
    information). 
    
    In general, a user-specified bounding function must accept
    two arguments: the candidate to be bounded and the keyword
    argument dictionary. Typically, the signature of such a 
    function would be the following::
    
        bounded_candidate = bounding_function(candidate, args)
        
    This function should return the resulting candidate after 
    bounding has been performed.
    
    Public Attributes:
    - *alleles_list* -- list of alleles
    - *bounds_list* -- list of bounds
    - *allele_types* -- list of types of alleles
    
    """
    def __init__(self, alleles_list=None, bounds_list=None, allele_types=None):
        self.alleles_list = alleles_list
        self.bounds_list = bounds_list
        self.allele_types = allele_types
        if len(self.alleles_list) != len(self.bounds_list):
            raise ValueError("The alleles list and bounds list must be the same length.")
        if len(self.alleles_list) != len(self.allele_types):
            raise ValueError("The alleles list and allele types list must be the same length.")

    def __call__(self, candidate, args):
        # The default would be to leave the candidate alone
        # unless allels tyes, list and bounds are defined.
        if self.alleles_list is None or self.bounds_list is None or self.allele_types is None:
            return candidate
        else:
            bounded_candidate = copy.copy(candidate)

            for i, alleles in enumerate(self.alleles_list):
                if self.allele_types[i] == 'value_coded':
                    bounded_candidate[alleles] = max(min(candidate[alleles], self.bounds_list[i][1]), self.bounds_list[i][0])
                elif self.allele_types[i] == 'gray_code':
                    dec = grayToDecimal(candidate[alleles[0]:alleles[1]])
                    bounded_dec = max(min(dec, self.bounds_list[i][1]), self.bounds_list[i][0])
                    bounded_gray = grayCode(bounded_dec, alleles[1] - alleles[0])
                    bounded_candidate[alleles[0]:alleles[1]] = bounded_gray
                elif self.allele_types[i] == 'bynary_code':
                    dec = int(''.join(str(c) for c in candidate[alleles[0]:alleles[1]]), 2)
                    bounded_dec = max(min(dec, self.bounds_list[i][1]), self.bounds_list[i][0])
                    bounded_binary = bin(bounded_dec)[2:]
                    bounded_candidate[alleles[0]:alleles[1]] = list(map(int, bounded_binary))
                elif self.allele_types[i] == 'value_coded_discrete':
                    candidate[alleles] = floor(candidate[alleles])
                    bounded_candidate[alleles] = max(min(candidate[alleles], self.bounds_list[i][1]), self.bounds_list[i][0])

            return bounded_candidate
        
