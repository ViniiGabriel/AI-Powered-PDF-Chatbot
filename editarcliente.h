#ifndef EDITARCLIENTE_H
#define EDITARCLIENTE_H

#include <QDialog>

namespace Ui {
class editarCliente;
}

class editarCliente : public QDialog
{
    Q_OBJECT

public:
    explicit editarCliente(QWidget *parent = nullptr, int id_cliente=0);
    ~editarCliente();

private slots:
    void on_radio_corrente_clicked();

    void on_radio_especial_clicked();

    void on_btn_gravar_clicked();

private:
    Ui::editarCliente *ui;
};

#endif // EDITARCLIENTE_H
