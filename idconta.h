#ifndef IDCONTA_H
#define IDCONTA_H

#include <QDialog>

namespace Ui {
class idConta;
}

class idConta : public QDialog
{
    Q_OBJECT

public:
    explicit idConta(QWidget *parent = nullptr);
    ~idConta();

private:
    Ui::idConta *ui;
};

#endif // IDCONTA_H
