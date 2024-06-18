#ifndef LISTACLIENTES_H
#define LISTACLIENTES_H

#include <QDialog>

namespace Ui {
class listaClientes;
}

class listaClientes : public QDialog
{
    Q_OBJECT

public:
    explicit listaClientes(QWidget *parent = nullptr);
    ~listaClientes();

private slots:
    void on_btn_excluir_clicked();

    void on_btn_editar_clicked();

private:
    Ui::listaClientes *ui;
};

#endif // LISTACLIENTES_H
