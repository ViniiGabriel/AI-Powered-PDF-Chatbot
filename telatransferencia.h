#ifndef TELATRANSFERENCIA_H
#define TELATRANSFERENCIA_H

#include <QDialog>
#include <QtSql>
#include <QMessageBox>

namespace Ui {
class telaTransferencia;
}

class telaTransferencia : public QDialog
{
    Q_OBJECT

public:
    explicit telaTransferencia(QWidget *parent = nullptr, int id=0);
    ~telaTransferencia();

private slots:
    void on_pushButton_clicked();

private:
    Ui::telaTransferencia *ui;
    int m_id;
};

#endif // TELATRANSFERENCIA_H
