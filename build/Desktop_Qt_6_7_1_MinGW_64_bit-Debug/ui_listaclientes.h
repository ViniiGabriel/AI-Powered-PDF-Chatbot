/********************************************************************************
** Form generated from reading UI file 'listaclientes.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LISTACLIENTES_H
#define UI_LISTACLIENTES_H

#include <QtCore/QVariant>
#include <QtWidgets/QAbstractButton>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_listaClientes
{
public:
    QDialogButtonBox *buttonBox;
    QLabel *label;
    QTableWidget *tb_lista;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QLineEdit *txt_nome;
    QPushButton *btn_pesquisa;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *btn_excluir;
    QPushButton *btn_editar;

    void setupUi(QDialog *listaClientes)
    {
        if (listaClientes->objectName().isEmpty())
            listaClientes->setObjectName("listaClientes");
        listaClientes->resize(800, 600);
        buttonBox = new QDialogButtonBox(listaClientes);
        buttonBox->setObjectName("buttonBox");
        buttonBox->setGeometry(QRect(440, 560, 351, 31));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        label = new QLabel(listaClientes);
        label->setObjectName("label");
        label->setGeometry(QRect(10, -10, 351, 71));
        QFont font;
        font.setFamilies({QString::fromUtf8("ADAM.CG PRO")});
        font.setPointSize(28);
        label->setFont(font);
        tb_lista = new QTableWidget(listaClientes);
        tb_lista->setObjectName("tb_lista");
        tb_lista->setGeometry(QRect(10, 100, 771, 371));
        layoutWidget = new QWidget(listaClientes);
        layoutWidget->setObjectName("layoutWidget");
        layoutWidget->setGeometry(QRect(11, 61, 561, 31));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        label_2 = new QLabel(layoutWidget);
        label_2->setObjectName("label_2");

        horizontalLayout->addWidget(label_2);

        txt_nome = new QLineEdit(layoutWidget);
        txt_nome->setObjectName("txt_nome");

        horizontalLayout->addWidget(txt_nome);

        btn_pesquisa = new QPushButton(layoutWidget);
        btn_pesquisa->setObjectName("btn_pesquisa");

        horizontalLayout->addWidget(btn_pesquisa);

        layoutWidget1 = new QWidget(listaClientes);
        layoutWidget1->setObjectName("layoutWidget1");
        layoutWidget1->setGeometry(QRect(10, 490, 169, 31));
        horizontalLayout_2 = new QHBoxLayout(layoutWidget1);
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        btn_excluir = new QPushButton(layoutWidget1);
        btn_excluir->setObjectName("btn_excluir");

        horizontalLayout_2->addWidget(btn_excluir);

        btn_editar = new QPushButton(layoutWidget1);
        btn_editar->setObjectName("btn_editar");

        horizontalLayout_2->addWidget(btn_editar);


        retranslateUi(listaClientes);
        QObject::connect(buttonBox, &QDialogButtonBox::accepted, listaClientes, qOverload<>(&QDialog::accept));
        QObject::connect(buttonBox, &QDialogButtonBox::rejected, listaClientes, qOverload<>(&QDialog::reject));

        QMetaObject::connectSlotsByName(listaClientes);
    } // setupUi

    void retranslateUi(QDialog *listaClientes)
    {
        listaClientes->setWindowTitle(QCoreApplication::translate("listaClientes", "Lista de Clientes", nullptr));
        label->setText(QCoreApplication::translate("listaClientes", "Lista de Clientes", nullptr));
        label_2->setText(QCoreApplication::translate("listaClientes", "Pesquisar nome:", nullptr));
        btn_pesquisa->setText(QCoreApplication::translate("listaClientes", "Pesquisa", nullptr));
        btn_excluir->setText(QCoreApplication::translate("listaClientes", "Excluir", nullptr));
        btn_editar->setText(QCoreApplication::translate("listaClientes", "Editar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class listaClientes: public Ui_listaClientes {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LISTACLIENTES_H
