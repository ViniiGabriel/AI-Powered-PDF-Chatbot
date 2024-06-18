#ifndef TELAIDCONTA_H
#define TELAIDCONTA_H

#include <QDialog>

namespace Ui {
class telaIdConta;
}

class telaIdConta : public QDialog
{
    Q_OBJECT

public:
    explicit telaIdConta(QWidget *parent = nullptr);
    ~telaIdConta();

private slots:
    void on_pushButton_clicked();

private:
    Ui::telaIdConta *ui;
};

#endif // TELAIDCONTA_H
