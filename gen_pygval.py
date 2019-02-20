import ModuleWrapper

def main():
    mw = ModuleWrapper.ModuleWrapper("pygval")
    mw.doc = "Module implements gval library wrappers."
    mw.addHeader("#include <GVal.h>")

    c = mw.addClass("GVal")
    c.addMethod("setString", "void(String)", "setString(x)")

    r = mw.emit()
    print(r)

main()
