#include "editarcliente.h"
#include "ui_editarcliente.h"
#include <QtSql>
#include <QMessageBox>

static int id;

editarCliente::editarCliente(QWidget *parent, int id_cliente)
    : QDialog(parent)
    , ui(new Ui::editarCliente)
{
    ui->setupUi(this);
    id = id_cliente;
    ui->txt_limite->setEnabled(false);
    ui->label_limite->setEnabled(false);

    QSqlQuery query;
    query.prepare("select * from tb_clientes where id="+QString::number(id_cliente));
    if(query.exec())
    {
        query.first();
        ui->txt_id->setText(QString::number(id_cliente));
        ui->txt_nome->setText(query.value(1).toString());
        ui->txt_endereco->setText(query.value(3).toString());
        ui->txt_profissao->setText(query.value(2).toString());
        ui->txt_renda->setText(query.value(4).toString());
        ui->txt_limite->setText(query.value(6).toString());

        QString tipoConta = query.value(5).toString();
        if(tipoConta == "corrente")
        {
            ui->radio_corrente->setChecked(true);
            ui->txt_limite->setEnabled(false);
            ui->label_limite->setEnabled(false);
        } else if(tipoConta == "especial")
        {
            ui->radio_especial->setChecked(true);
            ui->txt_limite->setEnabled(true);
            ui->label_limite->setEnabled(true);
        }

    }else{
        QMessageBox::warning(this,"ERRO","Erro ao selecionar contato.");
    }

}

editarCliente::~editarCliente()
{
    delete ui;
}

void editarCliente::on_radio_corrente_clicked()
{
    ui->txt_limite->setEnabled(false);
    ui->label_limite->setEnabled(false);
}


void editarCliente::on_radio_especial_clicked()
{
    ui->txt_limite->setEnabled(true);
    ui->label_limite->setEnabled(true);
}


void editarCliente::on_btn_gravar_clicked()
{
    QString nome = ui->txt_nome->text();
    QString endereco = ui->txt_endereco->text();
    QString profissao = ui->txt_profissao->text();
    QString renda = ui->txt_renda->text();
    QString tipoConta;
    QString limite;
    if(ui->radio_corrente->isChecked())
    {
        tipoConta = "corrente";
        limite = "0";

    } else if(ui->radio_especial->isChecked())
    {
        tipoConta = "especial";
        limite = ui->txt_limite->text();
    }

    QSqlQuery query;
    query.prepare("update tb_clientes set nome='"+nome+"',endereco='"+endereco+"',profissao='"+profissao+"',renda='"+renda+"',tipoConta='"+tipoConta+"',"
    "limite='"+limite+"' where id="+QString::number(id));
    if(query.exec()){
        this->close();
    } else {
        QMessageBox::warning(this,"ERRO","Erro ao atualizar contato");
    }
}

