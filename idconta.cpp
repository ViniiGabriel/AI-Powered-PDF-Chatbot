#include "idconta.h"
#include "ui_idconta.h"

idConta::idConta(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::idConta)
{
    ui->setupUi(this);
}

idConta::~idConta()
{
    delete ui;
}
