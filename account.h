#ifndef ACCOUNT_H
#define ACCOUNT_H

#include "client.h"

class Account
{
protected:
    Client* m_client;
    float m_balance;

public:
    Account(Client *client, float balance) : m_client(client), m_balance(balance) {}
    virtual ~Account() = 0;
    virtual bool Withdraw(float value) = 0;
    virtual bool Deposit(float value) = 0;
    virtual bool Transfer(Account *client, float value) = 0;

};

inline Account::~Account() {}

#endif // ACCOUNT_H
