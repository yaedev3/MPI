# Differences between FORTRAN 90 and FORTRAN 2008

## FORTRAN 95

1. Pointer initialization - You can now specify default initialization for a pointer.
2. Automatic deallocation of allocatable arrays.
3. Enhanced CEILING and FLOOR intrinsic functions
4. The FORALL statement and construct
5. PURE user-defined procedures
6. ELEMENTAL user-defined procedures
7. CPU_TIME intrinsic subroutine

## FORTRAN 2003

1.  Procedure pointers - A pointer or pointer component may be a procedure pointer.
2. Finalization - A derived type may have ‘final’ subroutines bound to it.
3. The PASS attribute
4. Procedures bound to a type as operators - A procedure may be bound to a type as an operator or a defined assignment.
5. Overriding a type-bound procedure - A specific procedure bound by name is permitted to have the name and attributes of a procedure bound to the parent. 
6. Enumerations - An enumeration is a set of integer constants (enumerators) that is appropriate for interoperating with C.
7. Polymorphic entities - A polymorphic entity is declared to be of a certain type by using the CLASS keyword in place of the TYPE keyword and is able to take this type or any of its extensions during execution.
8. The allocate statement - The allocatable attribute is no longer restricted to arrays and a source variable may be specified to provide values for deferred type parameters and an initial value for the object itself
9. Renaming operators on the USE statement.
10. Pointer assignment - Pointer assignment for arrays has been extended to allow lower bounds to be specified.
11. Pointer INTENT.
12. The VOLATILE attribute.
13. The IMPORT statement.
14. Array constructor syntax.
15. Complex constants.
16. Recursive input/output - A recursive input/output statement is one that is executed while another input/output statement is in execution. 

## FORTRAN 2008

1. Coarrays - Coarrays are variables that can be directly accessed by another image.
2. Submodules - Submodules allow a module procedure to have its interface defined in a module while having the body of the procedure defined in a separate unit, a submodule.
3. Procedure pointers can point to an internal procedure
4. Segments and synchronisation
   1. SYNC ALL
   2. SYNC IMAGES
   3. SYNC IMAGES (*)
   4. SYNC MEMORY
   5. ALLOCATE or DEALLOCATE
   6. CRITICAL and END CRITICAL
   7. LOCK and UNLOCK
   8. MOVE_ALLOC intrinsic

## Referencias 

[Co-Array Fortran What is it? Why should you put it on BlueGene/L?](https://asc.llnl.gov/computing_resources/bluegenel/papers/numrich.pdf)

[Fortran 2008 Overview](https://www.nag.co.uk/nagware/np/r62_doc/nag_f2008.html)

[Fortran 2008](http://fortranwiki.org/fortran/show/Fortran+2008)

[The new features of Fortran 2008](https://wg5-fortran.org/N1801-N1850/N1828.pdf)

[The New Features of Fortran 2003](https://wg5-fortran.org/N1601-N1650/N1648.pdf)

[Fortran 95 Features](http://h30266.www3.hpe.com/odl/unix/progtool/cf95au56/lrm0008.htm)