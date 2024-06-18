QT       += core gui sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    cadastrocliente.cpp \
    consultacliente.cpp \
    editarcliente.cpp \
    listaclientes.cpp \
    main.cpp \
    teladeposito.cpp \
    telaidconta.cpp \
    telainicial.cpp \
    telasaque.cpp \
    telatransferencia.cpp

HEADERS += \
    account.h \
    cadastrocliente.h \
    client.h \
    consultacliente.h \
    contacorrente.h \
    contaespecial.h \
    editarcliente.h \
    listaclientes.h \
    teladeposito.h \
    telaidconta.h \
    telainicial.h \
    telasaque.h \
    telatransferencia.h

FORMS += \
    cadastrocliente.ui \
    consultacliente.ui \
    editarcliente.ui \
    listaclientes.ui \
    teladeposito.ui \
    telaidconta.ui \
    telainicial.ui \
    telasaque.ui \
    telatransferencia.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
