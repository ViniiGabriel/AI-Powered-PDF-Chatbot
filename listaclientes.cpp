#include "listaclientes.h"
#include "ui_listaclientes.h"
#include "editarcliente.h"
#include <QtSql>
#include <QMessageBox>

listaClientes::listaClientes(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::listaClientes)
{
    ui->setupUi(this);
    QSqlQuery query;
    query.prepare("select * from tb_clientes");
    if(query.exec())
    {
        int linha=0;
        ui->tb_lista->setColumnCount(8);
        while(query.next())
        {
            ui->tb_lista->insertRow(linha);
            ui->tb_lista->setItem(linha, 0, new QTableWidgetItem(query.value(0).toString()));
            ui->tb_lista->setItem(linha, 1, new QTableWidgetItem(query.value(1).toString()));
            ui->tb_lista->setItem(linha, 2, new QTableWidgetItem(query.value(2).toString()));
            ui->tb_lista->setItem(linha, 3, new QTableWidgetItem(query.value(3).toString()));
            ui->tb_lista->setItem(linha, 4, new QTableWidgetItem(query.value(4).toString()));
            ui->tb_lista->setItem(linha, 5, new QTableWidgetItem(query.value(5).toString()));
            ui->tb_lista->setItem(linha, 6, new QTableWidgetItem(query.value(7).toString()));
            ui->tb_lista->setItem(linha, 7, new QTableWidgetItem(query.value(6).toString()));
            ui->tb_lista->setRowHeight(linha, 20);
            linha++;
        }
        ui->tb_lista->setColumnWidth(0, 20);

        QStringList cabecalho = {"ID","Nome","Profissão","Endereço","Renda","Tipo de Conta","Saldo","Limite"};
        ui->tb_lista->setHorizontalHeaderLabels(cabecalho);
        ui->tb_lista->setEditTriggers(QAbstractItemView::NoEditTriggers);
        ui->tb_lista->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->tb_lista->selectRow(0);
        ui->tb_lista->verticalHeader()->setVisible(false);
        ui->tb_lista->setStyleSheet("QTableView {selection-background-color:blue}");

    } else {
        QMessageBox::warning(this,"ERRO","Erro ao pesquisar clientes");
    }
}

listaClientes::~listaClientes()
{
    delete ui;
}



void listaClientes::on_btn_excluir_clicked()
{
    int linha=ui->tb_lista->currentRow();
    int id=ui->tb_lista->item(linha, 0)->text().toInt();


    QSqlQuery query;
    query.prepare("delete from tb_clientes where id="+QString::number(id));
    if(query.exec())
    {
        ui->tb_lista->removeRow(linha);
        QMessageBox::information(this,"SUCESSO","Cliente excluido com sucesso");
    } else {
        QMessageBox::warning(this, "ERRO", "Erro ao excluir cliente");
    }

}


void listaClientes::on_btn_editar_clicked()
{
    if(ui->tb_lista->rowCount() >= 0|| ui->tb_lista->currentRow() <= ui->tb_lista->rowCount()){
    int linha = ui->tb_lista->currentRow();
    int id = ui->tb_lista->item(linha,0)->text().toInt();
    editarCliente editar(this, id);
    editar.setModal(true);
    editar.exec();

    //Carregar contatos novamente
    QSqlQuery query;
    query.prepare("select * from tb_clientes where id="+QString::number(id));
    if(query.exec())
    {
        query.first();
        ui->tb_lista->setItem(linha, 1, new QTableWidgetItem(query.value(1).toString()));
        ui->tb_lista->setItem(linha, 2, new QTableWidgetItem(query.value(2).toString()));
        ui->tb_lista->setItem(linha, 3, new QTableWidgetItem(query.value(3).toString()));
        ui->tb_lista->setItem(linha, 4, new QTableWidgetItem(query.value(4).toString()));
        ui->tb_lista->setItem(linha, 5, new QTableWidgetItem(query.value(5).toString()));
        ui->tb_lista->setItem(linha, 7, new QTableWidgetItem(query.value(6).toString()));
    } else {
        QMessageBox::warning(this,"ERRO","Erro ao carregar informações");
    }
    } else {
        QMessageBox::information(this,"AVISO","Selecione um cliente");
    }
}


void listaClientes::on_btn_pesquisa_clicked()
{
    ui->tb_lista->clear();
}

