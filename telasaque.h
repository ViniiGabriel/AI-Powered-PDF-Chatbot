#ifndef TELASAQUE_H
#define TELASAQUE_H

#include <QDialog>

namespace Ui {
class telaSaque;
}

class telaSaque : public QDialog
{
    Q_OBJECT

public:
    explicit telaSaque(QWidget *parent = nullptr, int id=0);
    ~telaSaque();

private slots:
    void on_btn_saque_clicked();

private:
    Ui::telaSaque *ui;
    int m_id;
};

#endif // TELASAQUE_H
