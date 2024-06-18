#ifndef CONSULTACLIENTE_H
#define CONSULTACLIENTE_H

#include <QDialog>
#include <QtSql>
#include <QMessageBox>

namespace Ui {
class consultaCliente;
}

class consultaCliente : public QDialog
{
    Q_OBJECT

public:
    explicit consultaCliente(QWidget *parent = nullptr, int id_cliente=0);
    ~consultaCliente();

private slots:
    void on_btn_deposito_clicked();

    void on_btn_saque_clicked();

    void on_btn_transferencia_clicked();

private:
    Ui::consultaCliente *ui;
    int id;
};

#endif // CONSULTACLIENTE_H
