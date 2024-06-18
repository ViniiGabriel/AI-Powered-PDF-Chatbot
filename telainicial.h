#ifndef TELAINICIAL_H
#define TELAINICIAL_H

#include <QMainWindow>
#include <QtSql>
#include <QDebug>
#include <QFileInfo>

QT_BEGIN_NAMESPACE
namespace Ui {
class telaInicial;
}
QT_END_NAMESPACE

class telaInicial : public QMainWindow
{
    Q_OBJECT

public:
    telaInicial(QWidget *parent = nullptr);
    ~telaInicial();

private slots:
    void on_btn_cadastro_clicked();

    void on_btn_lista_clicked();

    void on_btn_consulta_clicked();

private:
    Ui::telaInicial *ui;
};
#endif // TELAINICIAL_H
