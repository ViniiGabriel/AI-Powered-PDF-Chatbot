/********************************************************************************
** Form generated from reading UI file 'telainicial.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TELAINICIAL_H
#define UI_TELAINICIAL_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_telaInicial
{
public:
    QWidget *centralwidget;
    QLabel *label;
    QPushButton *btn_cadastro;
    QPushButton *btn_consulta;
    QPushButton *btn_lista;
    QLabel *label_2;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *telaInicial)
    {
        if (telaInicial->objectName().isEmpty())
            telaInicial->setObjectName("telaInicial");
        telaInicial->resize(800, 601);
        QIcon icon(QIcon::fromTheme(QString::fromUtf8("application-exit")));
        telaInicial->setWindowIcon(icon);
        centralwidget = new QWidget(telaInicial);
        centralwidget->setObjectName("centralwidget");
        label = new QLabel(centralwidget);
        label->setObjectName("label");
        label->setGeometry(QRect(100, 10, 551, 91));
        QFont font;
        font.setFamilies({QString::fromUtf8("Airside Sans")});
        font.setPointSize(48);
        label->setFont(font);
        label->setAlignment(Qt::AlignCenter);
        btn_cadastro = new QPushButton(centralwidget);
        btn_cadastro->setObjectName("btn_cadastro");
        btn_cadastro->setGeometry(QRect(270, 130, 221, 71));
        btn_consulta = new QPushButton(centralwidget);
        btn_consulta->setObjectName("btn_consulta");
        btn_consulta->setGeometry(QRect(270, 230, 221, 71));
        btn_lista = new QPushButton(centralwidget);
        btn_lista->setObjectName("btn_lista");
        btn_lista->setGeometry(QRect(270, 340, 221, 71));
        label_2 = new QLabel(centralwidget);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(10, 520, 641, 31));
        QFont font1;
        font1.setPointSize(12);
        label_2->setFont(font1);
        telaInicial->setCentralWidget(centralwidget);
        menubar = new QMenuBar(telaInicial);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 800, 25));
        telaInicial->setMenuBar(menubar);
        statusbar = new QStatusBar(telaInicial);
        statusbar->setObjectName("statusbar");
        telaInicial->setStatusBar(statusbar);

        retranslateUi(telaInicial);

        QMetaObject::connectSlotsByName(telaInicial);
    } // setupUi

    void retranslateUi(QMainWindow *telaInicial)
    {
        telaInicial->setWindowTitle(QCoreApplication::translate("telaInicial", "Banco UFSC", nullptr));
        label->setText(QCoreApplication::translate("telaInicial", "BANCO UFSC", nullptr));
        btn_cadastro->setText(QCoreApplication::translate("telaInicial", "CADASTRAR CLIENTE", nullptr));
        btn_consulta->setText(QCoreApplication::translate("telaInicial", "CONSULTAR CLIENTE", nullptr));
        btn_lista->setText(QCoreApplication::translate("telaInicial", "LISTA DE CLIENTES", nullptr));
        label_2->setText(QCoreApplication::translate("telaInicial", "Feito por: Bruno Arenhart Rodrigues e Vin\303\255cius Domingos Gabriel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class telaInicial: public Ui_telaInicial {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TELAINICIAL_H
