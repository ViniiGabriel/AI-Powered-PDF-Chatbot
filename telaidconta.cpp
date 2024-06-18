#include "telaidconta.h"
#include "ui_telaidconta.h"
#include "consultacliente.h"

telaIdConta::telaIdConta(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::telaIdConta)
{
    ui->setupUi(this);
}

telaIdConta::~telaIdConta()
{
    delete ui;
}

void telaIdConta::on_pushButton_clicked()
{
    int id = ui->txt_id->text().toInt();
    consultaCliente consulta(NULL, id);
    consulta.setModal(true);
    this->close();
    consulta.exec();
}

