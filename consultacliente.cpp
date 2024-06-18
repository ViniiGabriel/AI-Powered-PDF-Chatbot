#include "consultacliente.h"
#include "ui_consultacliente.h"
#include "teladeposito.h"
#include "telasaque.h"
#include "telatransferencia.h"

consultaCliente::consultaCliente(QWidget *parent, int id_cliente)
    : QDialog(parent)
    , ui(new Ui::consultaCliente)
    , id(id_cliente)
{
    ui->setupUi(this);

    QSqlQuery query;
    query.prepare("select * from tb_clientes where id="+QString::number(id));
    if(query.exec()){
        query.first();
        ui->txt_id->setText(QString::number(id_cliente));
        ui->txt_nome->setText(query.value(1).toString());
        ui->txt_endereco->setText(query.value(3).toString());
        ui->txt_profissao->setText(query.value(2).toString());
        ui->txt_renda->setText(query.value(4).toString());
        ui->txt_limite->setText(query.value(6).toString());
        ui->txt_saldo->setText(query.value(7).toString());
        ui->txt_limiteDisp->setText(query.value(8).toString());
        QString tipoConta = query.value(5).toString();
        ui->txt_tipoConta->setText(tipoConta);
        if(tipoConta == "corrente")
        {
            ui->txt_limite->setVisible(false);
            ui->label_limite->setVisible(false);
            ui->label_rslimite->setVisible(false);
            ui->txt_limiteDisp->setVisible(false);
            ui->label_limiteDisp->setVisible(false);
            ui->label_rslimiteDisp->setVisible(false);

        } else if(tipoConta == "especial")
        {
            ui->txt_limite->setVisible(true);
            ui->label_limite->setVisible(true);
            ui->label_rslimite->setVisible(true);
            ui->txt_limiteDisp->setVisible(true);
            ui->label_limiteDisp->setVisible(true);
            ui->label_rslimiteDisp->setVisible(true);
        }
    } else {
        QMessageBox::warning(this, "ERRO", "Erro ao consultar informações");
    }
}

consultaCliente::~consultaCliente()
{
    delete ui;
}

void consultaCliente::on_btn_deposito_clicked()
{
    telaDeposito tela(NULL, id);
    tela.setModal(true);
    tela.exec();
    QSqlQuery query;
    query.prepare("select * from tb_clientes where id="+QString::number(id));
    if(query.exec()){
        query.first();
        ui->txt_saldo->setText(query.value(7).toString());
        ui->txt_limite->setText(query.value(6).toString());
        ui->txt_limiteDisp->setText(query.value(8).toString());
    }
}


void consultaCliente::on_btn_saque_clicked()
{
    telaSaque tela(NULL, id);
    tela.setModal(true);
    tela.exec();
    QSqlQuery query;
    query.prepare("select * from tb_clientes where id="+QString::number(id));
    if(query.exec()){
        query.first();
        ui->txt_saldo->setText(query.value(7).toString());
        ui->txt_limite->setText(query.value(6).toString());
        ui->txt_limiteDisp->setText(query.value(8).toString());
    }
}


void consultaCliente::on_btn_transferencia_clicked()
{
    telaTransferencia tela(NULL, id);
    tela.setModal(true);
    tela.exec();
    QSqlQuery query;
    query.prepare("select * from tb_clientes where id="+QString::number(id));
    if(query.exec()){
        query.first();
        ui->txt_saldo->setText(query.value(7).toString());
        ui->txt_limite->setText(query.value(6).toString());
        ui->txt_limiteDisp->setText(query.value(8).toString());
    }
}

