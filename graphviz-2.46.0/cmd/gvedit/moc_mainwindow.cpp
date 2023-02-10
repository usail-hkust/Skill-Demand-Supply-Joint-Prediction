/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "mainwindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CMainWindow_t {
    QByteArrayData data[21];
    char stringdata0[205];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CMainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CMainWindow_t qt_meta_stringdata_CMainWindow = {
    {
QT_MOC_LITERAL(0, 0, 11), // "CMainWindow"
QT_MOC_LITERAL(1, 12, 12), // "slotSettings"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 7), // "slotRun"
QT_MOC_LITERAL(4, 34, 9), // "MdiChild*"
QT_MOC_LITERAL(5, 44, 1), // "m"
QT_MOC_LITERAL(6, 46, 7), // "slotNew"
QT_MOC_LITERAL(7, 54, 8), // "slotOpen"
QT_MOC_LITERAL(8, 63, 8), // "slotSave"
QT_MOC_LITERAL(9, 72, 10), // "slotSaveAs"
QT_MOC_LITERAL(10, 83, 7), // "slotCut"
QT_MOC_LITERAL(11, 91, 8), // "slotCopy"
QT_MOC_LITERAL(12, 100, 9), // "slotPaste"
QT_MOC_LITERAL(13, 110, 9), // "slotAbout"
QT_MOC_LITERAL(14, 120, 16), // "slotRefreshMenus"
QT_MOC_LITERAL(15, 137, 10), // "slotNewLog"
QT_MOC_LITERAL(16, 148, 11), // "slotSaveLog"
QT_MOC_LITERAL(17, 160, 14), // "createMdiChild"
QT_MOC_LITERAL(18, 175, 13), // "activateChild"
QT_MOC_LITERAL(19, 189, 8), // "QWidget*"
QT_MOC_LITERAL(20, 198, 6) // "window"

    },
    "CMainWindow\0slotSettings\0\0slotRun\0"
    "MdiChild*\0m\0slotNew\0slotOpen\0slotSave\0"
    "slotSaveAs\0slotCut\0slotCopy\0slotPaste\0"
    "slotAbout\0slotRefreshMenus\0slotNewLog\0"
    "slotSaveLog\0createMdiChild\0activateChild\0"
    "QWidget*\0window"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CMainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   94,    2, 0x08 /* Private */,
       3,    1,   95,    2, 0x08 /* Private */,
       3,    0,   98,    2, 0x28 /* Private | MethodCloned */,
       6,    0,   99,    2, 0x08 /* Private */,
       7,    0,  100,    2, 0x08 /* Private */,
       8,    0,  101,    2, 0x08 /* Private */,
       9,    0,  102,    2, 0x08 /* Private */,
      10,    0,  103,    2, 0x08 /* Private */,
      11,    0,  104,    2, 0x08 /* Private */,
      12,    0,  105,    2, 0x08 /* Private */,
      13,    0,  106,    2, 0x08 /* Private */,
      14,    0,  107,    2, 0x08 /* Private */,
      15,    0,  108,    2, 0x08 /* Private */,
      16,    0,  109,    2, 0x08 /* Private */,
      17,    0,  110,    2, 0x08 /* Private */,
      18,    1,  111,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    0x80000000 | 4,
    QMetaType::Void, 0x80000000 | 19,   20,

       0        // eod
};

void CMainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        CMainWindow *_t = static_cast<CMainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->slotSettings(); break;
        case 1: _t->slotRun((*reinterpret_cast< MdiChild*(*)>(_a[1]))); break;
        case 2: _t->slotRun(); break;
        case 3: _t->slotNew(); break;
        case 4: _t->slotOpen(); break;
        case 5: _t->slotSave(); break;
        case 6: _t->slotSaveAs(); break;
        case 7: _t->slotCut(); break;
        case 8: _t->slotCopy(); break;
        case 9: _t->slotPaste(); break;
        case 10: _t->slotAbout(); break;
        case 11: _t->slotRefreshMenus(); break;
        case 12: _t->slotNewLog(); break;
        case 13: _t->slotSaveLog(); break;
        case 14: { MdiChild* _r = _t->createMdiChild();
            if (_a[0]) *reinterpret_cast< MdiChild**>(_a[0]) = std::move(_r); }  break;
        case 15: _t->activateChild((*reinterpret_cast< QWidget*(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject CMainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_CMainWindow.data,
      qt_meta_data_CMainWindow,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *CMainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CMainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CMainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int CMainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 16;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
