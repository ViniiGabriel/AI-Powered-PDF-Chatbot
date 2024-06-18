/********************************************************************************
** Form generated from reading UI file 'teladeposito.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TELADEPOSITO_H
#define UI_TELADEPOSITO_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_telaDeposito
{
public:
    QLabel *label;
    QLineEdit *txt_valor;
    QPushButton *btn_depositar;

    void setupUi(QDialog *telaDeposito)
    {
        if (telaDeposito->objectName().isEmpty())
            telaDeposito->setObjectName("telaDeposito");
        telaDeposito->resize(403, 275);
        label = new QLabel(telaDeposito);
        label->setObjectName("label");
        label->setGeometry(QRect(30, 30, 341, 41));
        QFont font;
        font.setPointSize(23);
        label->setFont(font);
        txt_valor = new QLineEdit(telaDeposito);
        txt_valor->setObjectName("txt_valor");
        txt_valor->setGeometry(QRect(60, 90, 281, 61));
        btn_depositar = new QPushButton(telaDeposito);
        btn_depositar->setObjectName("btn_depositar");
        btn_depositar->setGeometry(QRect(60, 170, 281, 61));

        retranslateUi(telaDeposito);

        QMetaObject::connectSlotsByName(telaDeposito);
    } // setupUi

    void retranslateUi(QDialog *telaDeposito)
    {
        telaDeposito->setWindowTitle(QCoreApplication::translate("telaDeposito", "Dialog", nullptr));
        label->setText(QCoreApplication::translate("telaDeposito", "Valor para depositar:", nullptr));
        btn_depositar->setText(QCoreApplication::translate("telaDeposito", "Depositar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class telaDeposito: public Ui_telaDeposito {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TELADEPOSITO_H
