#include "teladeposito.h"
#include "ui_teladeposito.h"
#include "contaespecial.h"
#include "contacorrente.h"
#include <QDebug>

using namespace std;

telaDeposito::telaDeposito(QWidget *parent, int id)
    : QDialog(parent)
    , ui(new Ui::telaDeposito)
    , m_id(id)
{
    ui->setupUi(this);
}

telaDeposito::~telaDeposito()
{
    delete ui;
}



void telaDeposito::on_btn_depositar_clicked()
{
    query.prepare("select * from tb_clientes where id="+QString::number(m_id));
    if(query.exec()){
        query.first();
        QString nome = query.value(1).toString();
        QString endereco = query.value(3).toString();
        QString profissao = query.value(2).toString();
        float renda = query.value(4).toFloat();
        float saldo = query.value(7).toFloat();
        float limite = query.value(6).toFloat();
        float limiteDisp = query.value(8).toFloat();
        float valor = ui->txt_valor->text().toFloat();
        Client *cliente = new Client(nome.toStdString(),endereco.toStdString(),profissao.toStdString(),renda);
        if(query.value(5) == "corrente")
        {
            ContaCorrente conta(cliente, saldo, m_id);
            if(!conta.Deposit(valor))
            {
                QMessageBox::warning(this, "Erro", "Erro ao depositar");
            } else {
                QMessageBox::information(this, "Sucesso", "Deposito feito com sucesso");
                this->close();
            }

        } else if (query.value(5) == "especial") {
            ContaEspecial conta(cliente, saldo, limite, limiteDisp, m_id);
            if(!conta.Deposit(valor))
            {
                QMessageBox::warning(this, "Erro", "Erro ao depositar");
            } else {
                QMessageBox::information(this, "Sucesso", "Deposito feito com sucesso");
                this->close();
            }
        }
        delete cliente;
    } else {
        QMessageBox::warning(this,"ERRO","Erro no deposito");
    }

}

