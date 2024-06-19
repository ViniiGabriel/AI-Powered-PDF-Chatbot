/********************************************************************************
** Form generated from reading UI file 'telatransferencia.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TELATRANSFERENCIA_H
#define UI_TELATRANSFERENCIA_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_telaTransferencia
{
public:
    QLabel *label;
    QLineEdit *txt_valor;
    QLabel *label_2;
    QLineEdit *txt_id_destinatario;
    QPushButton *pushButton;

    void setupUi(QDialog *telaTransferencia)
    {
        if (telaTransferencia->objectName().isEmpty())
            telaTransferencia->setObjectName("telaTransferencia");
        telaTransferencia->resize(632, 418);
        label = new QLabel(telaTransferencia);
        label->setObjectName("label");
        label->setGeometry(QRect(140, 30, 321, 71));
        QFont font;
        font.setPointSize(24);
        label->setFont(font);
        txt_valor = new QLineEdit(telaTransferencia);
        txt_valor->setObjectName("txt_valor");
        txt_valor->setGeometry(QRect(140, 100, 321, 81));
        label_2 = new QLabel(telaTransferencia);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(170, 180, 271, 71));
        label_2->setFont(font);
        txt_id_destinatario = new QLineEdit(telaTransferencia);
        txt_id_destinatario->setObjectName("txt_id_destinatario");
        txt_id_destinatario->setGeometry(QRect(140, 250, 321, 81));
        pushButton = new QPushButton(telaTransferencia);
        pushButton->setObjectName("pushButton");
        pushButton->setGeometry(QRect(140, 350, 321, 51));
        QFont font1;
        font1.setPointSize(15);
        pushButton->setFont(font1);

        retranslateUi(telaTransferencia);

        QMetaObject::connectSlotsByName(telaTransferencia);
    } // setupUi

    void retranslateUi(QDialog *telaTransferencia)
    {
        telaTransferencia->setWindowTitle(QCoreApplication::translate("telaTransferencia", "Transfer\303\252ncia", nullptr));
        label->setText(QCoreApplication::translate("telaTransferencia", "Valor da Transfer\303\252ncia:", nullptr));
        label_2->setText(QCoreApplication::translate("telaTransferencia", "ID Destinat\303\241rio", nullptr));
        pushButton->setText(QCoreApplication::translate("telaTransferencia", "Transferir", nullptr));
    } // retranslateUi

};

namespace Ui {
    class telaTransferencia: public Ui_telaTransferencia {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TELATRANSFERENCIA_H
