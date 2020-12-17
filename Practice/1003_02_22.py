#!/usr/bin/python3

class Base(object):
    Obj_Num = 0

    @staticmethod
    def static_fun():
        print("This is a static method to count object numbers: "+ str(Base.Obj_Num))
        pass

    @classmethod
    def class_fun(cls):
        print("This is a class method to count object numbers: "+ str(Base.Obj_Num))
        cls("tmp").bound_fun()
        pass

    def __init__(self, name):
        self.__Name = name
        Base.Obj_Num += 1
        print("Create 1 obj: Total number of Base objects is "+ str(Base.Obj_Num))

    def __del__(self):
        Base.Obj_Num -= 1
        print("Delete 1 obj: Total number of Base objects is "+ str(Base.Obj_Num))
    
    def bound_fun(self):
        print("This is a bounded function of object "+ self.__Name)

if __name__ == '__main__':
    Base.static_fun()
    print("")
    print(type(Base.static_fun))
    print("")
    Base.class_fun()
    print("")
    print(type(Base.class_fun))
    print("")

    ObjA = Base("A")
    print("")
    ObjB = Base("B")
    print("")
    print(type(ObjA.bound_fun))
    print("")

    ObjA.static_fun()
    print("")
    ObjB.static_fun()
    print("")

    ObjA.class_fun()
    print("")
    ObjB.class_fun()
    print("")

    ObjA.bound_fun()
    print("")
    ObjB.bound_fun()
    print("")

    del ObjA
    print("")
    del ObjB