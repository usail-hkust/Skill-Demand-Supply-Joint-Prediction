/****************************************************************************
** Meta object code from reading C++ file 'csettings.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "csettings.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'csettings.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CFrmSettings_t {
    QByteArrayData data[11];
    char stringdata0[103];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CFrmSettings_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CFrmSettings_t qt_meta_stringdata_CFrmSettings = {
    {
QT_MOC_LITERAL(0, 0, 12), // "CFrmSettings"
QT_MOC_LITERAL(1, 13, 10), // "outputSlot"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 7), // "addSlot"
QT_MOC_LITERAL(4, 33, 8), // "helpSlot"
QT_MOC_LITERAL(5, 42, 10), // "cancelSlot"
QT_MOC_LITERAL(6, 53, 6), // "okSlot"
QT_MOC_LITERAL(7, 60, 7), // "newSlot"
QT_MOC_LITERAL(8, 68, 8), // "openSlot"
QT_MOC_LITERAL(9, 77, 8), // "saveSlot"
QT_MOC_LITERAL(10, 86, 16) // "scopeChangedSlot"

    },
    "CFrmSettings\0outputSlot\0\0addSlot\0"
    "helpSlot\0cancelSlot\0okSlot\0newSlot\0"
    "openSlot\0saveSlot\0scopeChangedSlot"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CFrmSettings[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   59,    2, 0x08 /* Private */,
       3,    0,   60,    2, 0x08 /* Private */,
       4,    0,   61,    2, 0x08 /* Private */,
       5,    0,   62,    2, 0x08 /* Private */,
       6,    0,   63,    2, 0x08 /* Private */,
       7,    0,   64,    2, 0x08 /* Private */,
       8,    0,   65,    2, 0x08 /* Private */,
       9,    0,   66,    2, 0x08 /* Private */,
      10,    1,   67,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    2,

       0        // eod
};

void CFrmSettings::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        CFrmSettings *_t = static_cast<CFrmSettings *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->outputSlot(); break;
        case 1: _t->addSlot(); break;
        case 2: _t->helpSlot(); break;
        case 3: _t->cancelSlot(); break;
        case 4: _t->okSlot(); break;
        case 5: _t->newSlot(); break;
        case 6: _t->openSlot(); break;
        case 7: _t->saveSlot(); break;
        case 8: _t->scopeChangedSlot((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject CFrmSettings::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_CFrmSettings.data,
      qt_meta_data_CFrmSettings,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *CFrmSettings::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CFrmSettings::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CFrmSettings.stringdata0))
        return static_cast<void*>(this);
    return QDialog::qt_metacast(_clname);
}

int CFrmSettings::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
