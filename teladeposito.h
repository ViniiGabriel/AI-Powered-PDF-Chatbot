#ifndef TELADEPOSITO_H
#define TELADEPOSITO_H

#include <QDialog>
#include <QtSql>
#include <QMessageBox>

namespace Ui {
class telaDeposito;
}

class telaDeposito : public QDialog
{
    Q_OBJECT

public:
    explicit telaDeposito(QWidget *parent = nullptr, int id=0);
    ~telaDeposito();

private slots:
    void on_btn_depositar_clicked();

private:
    Ui::telaDeposito *ui;
    int m_id;
    QSqlQuery query;
};

#endif // TELADEPOSITO_H
