#ifndef CONTACORRENTE_H
#define CONTACORRENTE_H

#include "account.h"
#include <QtSql>

class ContaCorrente : public Account
{

private:
    int m_id;

public:
    ContaCorrente(Client *client, float balance = 0, int id=0) : Account(client, balance), m_id(id) {}
    ~ContaCorrente() {}

    bool Withdraw(float value)
    {
        if (value <= m_balance)
        {
            m_balance -= value;
            QSqlQuery query;
            query.prepare("update tb_clientes set saldo='"+QString::number(m_balance)+"' where id="+QString::number(m_id));
            if(query.exec()){
                return true;
            }
        }
        return false;
    }

    bool Deposit(float value)
    {

        QSqlQuery query;
        m_balance += value;
        query.prepare("update tb_clientes set saldo='"+QString::number(m_balance)+"' where id="+QString::number(m_id));
        if(query.exec()){
            return true;
        } else {
            return false;
        }
    }

    bool Transfer(Account *client, float value)
    {
        if (value <= m_balance)
        {
            Withdraw(value);
            client->Deposit(value);
            return true;
            } else {
                return false;
            }
        }
};

#endif // CONTACORRENTE_H
