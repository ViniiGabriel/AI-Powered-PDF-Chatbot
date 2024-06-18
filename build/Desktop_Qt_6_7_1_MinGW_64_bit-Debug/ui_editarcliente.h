/********************************************************************************
** Form generated from reading UI file 'editarcliente.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_EDITARCLIENTE_H
#define UI_EDITARCLIENTE_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_editarCliente
{
public:
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLabel *txt_id;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_limite;
    QLineEdit *txt_limite;
    QWidget *widget2;
    QVBoxLayout *verticalLayout;
    QLabel *label_6;
    QHBoxLayout *horizontalLayout_6;
    QRadioButton *radio_corrente;
    QRadioButton *radio_especial;
    QWidget *widget3;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_2;
    QLineEdit *txt_nome;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_3;
    QLineEdit *txt_profissao;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_4;
    QLineEdit *txt_endereco;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_5;
    QLineEdit *txt_renda;
    QWidget *widget4;
    QHBoxLayout *horizontalLayout_8;
    QPushButton *btn_gravar;
    QPushButton *pushButton_2;

    void setupUi(QDialog *editarCliente)
    {
        if (editarCliente->objectName().isEmpty())
            editarCliente->setObjectName("editarCliente");
        editarCliente->resize(745, 392);
        widget = new QWidget(editarCliente);
        widget->setObjectName("widget");
        widget->setGeometry(QRect(30, 20, 90, 22));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(widget);
        label->setObjectName("label");

        horizontalLayout->addWidget(label);

        txt_id = new QLabel(widget);
        txt_id->setObjectName("txt_id");

        horizontalLayout->addWidget(txt_id);

        widget1 = new QWidget(editarCliente);
        widget1->setObjectName("widget1");
        widget1->setGeometry(QRect(30, 270, 341, 30));
        horizontalLayout_7 = new QHBoxLayout(widget1);
        horizontalLayout_7->setObjectName("horizontalLayout_7");
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        label_limite = new QLabel(widget1);
        label_limite->setObjectName("label_limite");

        horizontalLayout_7->addWidget(label_limite);

        txt_limite = new QLineEdit(widget1);
        txt_limite->setObjectName("txt_limite");

        horizontalLayout_7->addWidget(txt_limite);

        widget2 = new QWidget(editarCliente);
        widget2->setObjectName("widget2");
        widget2->setGeometry(QRect(30, 200, 241, 57));
        verticalLayout = new QVBoxLayout(widget2);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        label_6 = new QLabel(widget2);
        label_6->setObjectName("label_6");

        verticalLayout->addWidget(label_6);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName("horizontalLayout_6");
        radio_corrente = new QRadioButton(widget2);
        radio_corrente->setObjectName("radio_corrente");

        horizontalLayout_6->addWidget(radio_corrente);

        radio_especial = new QRadioButton(widget2);
        radio_especial->setObjectName("radio_especial");

        horizontalLayout_6->addWidget(radio_especial);


        verticalLayout->addLayout(horizontalLayout_6);

        widget3 = new QWidget(editarCliente);
        widget3->setObjectName("widget3");
        widget3->setGeometry(QRect(30, 50, 691, 143));
        verticalLayout_2 = new QVBoxLayout(widget3);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        label_2 = new QLabel(widget3);
        label_2->setObjectName("label_2");

        horizontalLayout_2->addWidget(label_2);

        txt_nome = new QLineEdit(widget3);
        txt_nome->setObjectName("txt_nome");

        horizontalLayout_2->addWidget(txt_nome);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        label_3 = new QLabel(widget3);
        label_3->setObjectName("label_3");

        horizontalLayout_3->addWidget(label_3);

        txt_profissao = new QLineEdit(widget3);
        txt_profissao->setObjectName("txt_profissao");

        horizontalLayout_3->addWidget(txt_profissao);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        label_4 = new QLabel(widget3);
        label_4->setObjectName("label_4");

        horizontalLayout_4->addWidget(label_4);

        txt_endereco = new QLineEdit(widget3);
        txt_endereco->setObjectName("txt_endereco");

        horizontalLayout_4->addWidget(txt_endereco);


        verticalLayout_2->addLayout(horizontalLayout_4);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName("horizontalLayout_5");
        label_5 = new QLabel(widget3);
        label_5->setObjectName("label_5");

        horizontalLayout_5->addWidget(label_5);

        txt_renda = new QLineEdit(widget3);
        txt_renda->setObjectName("txt_renda");

        horizontalLayout_5->addWidget(txt_renda);


        verticalLayout_2->addLayout(horizontalLayout_5);

        widget4 = new QWidget(editarCliente);
        widget4->setObjectName("widget4");
        widget4->setGeometry(QRect(30, 320, 691, 31));
        horizontalLayout_8 = new QHBoxLayout(widget4);
        horizontalLayout_8->setObjectName("horizontalLayout_8");
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        btn_gravar = new QPushButton(widget4);
        btn_gravar->setObjectName("btn_gravar");

        horizontalLayout_8->addWidget(btn_gravar);

        pushButton_2 = new QPushButton(widget4);
        pushButton_2->setObjectName("pushButton_2");

        horizontalLayout_8->addWidget(pushButton_2);


        retranslateUi(editarCliente);

        QMetaObject::connectSlotsByName(editarCliente);
    } // setupUi

    void retranslateUi(QDialog *editarCliente)
    {
        editarCliente->setWindowTitle(QCoreApplication::translate("editarCliente", "Dialog", nullptr));
        label->setText(QCoreApplication::translate("editarCliente", "ID:", nullptr));
        txt_id->setText(QCoreApplication::translate("editarCliente", "TextLabel", nullptr));
        label_limite->setText(QCoreApplication::translate("editarCliente", "Limite:", nullptr));
        label_6->setText(QCoreApplication::translate("editarCliente", "Tipo de conta:", nullptr));
        radio_corrente->setText(QCoreApplication::translate("editarCliente", "Corrente", nullptr));
        radio_especial->setText(QCoreApplication::translate("editarCliente", "Especial", nullptr));
        label_2->setText(QCoreApplication::translate("editarCliente", "Nome:", nullptr));
        label_3->setText(QCoreApplication::translate("editarCliente", "Profiss\303\243o:", nullptr));
        label_4->setText(QCoreApplication::translate("editarCliente", "Endere\303\247o:", nullptr));
        label_5->setText(QCoreApplication::translate("editarCliente", "Renda:", nullptr));
        btn_gravar->setText(QCoreApplication::translate("editarCliente", "Gravar", nullptr));
        pushButton_2->setText(QCoreApplication::translate("editarCliente", "Cancelar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class editarCliente: public Ui_editarCliente {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_EDITARCLIENTE_H
