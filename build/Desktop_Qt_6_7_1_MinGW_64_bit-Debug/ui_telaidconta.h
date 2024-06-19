/********************************************************************************
** Form generated from reading UI file 'telaidconta.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TELAIDCONTA_H
#define UI_TELAIDCONTA_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_telaIdConta
{
public:
    QLabel *label;
    QPushButton *pushButton;
    QLineEdit *txt_id;

    void setupUi(QDialog *telaIdConta)
    {
        if (telaIdConta->objectName().isEmpty())
            telaIdConta->setObjectName("telaIdConta");
        telaIdConta->resize(399, 234);
        label = new QLabel(telaIdConta);
        label->setObjectName("label");
        label->setGeometry(QRect(120, 30, 171, 51));
        QFont font;
        font.setPointSize(18);
        label->setFont(font);
        pushButton = new QPushButton(telaIdConta);
        pushButton->setObjectName("pushButton");
        pushButton->setGeometry(QRect(110, 160, 161, 41));
        txt_id = new QLineEdit(telaIdConta);
        txt_id->setObjectName("txt_id");
        txt_id->setGeometry(QRect(50, 80, 291, 61));

        retranslateUi(telaIdConta);

        QMetaObject::connectSlotsByName(telaIdConta);
    } // setupUi

    void retranslateUi(QDialog *telaIdConta)
    {
        telaIdConta->setWindowTitle(QCoreApplication::translate("telaIdConta", "ID", nullptr));
        label->setText(QCoreApplication::translate("telaIdConta", "ID da conta:", nullptr));
        pushButton->setText(QCoreApplication::translate("telaIdConta", "Consultar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class telaIdConta: public Ui_telaIdConta {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TELAIDCONTA_H
