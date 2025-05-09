import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import {
    Button,
    Flex,
    Field,
    Input,
    VStack,
} from '@chakra-ui/react'

import { login } from '@/shared/api'

const LoginPage = () => {
    const [form, setForm] = useState<{ email: string; password: string }>({
        email: '',
        password: '',
    })
    const navigate = useNavigate()

    const handleLoginChange = (e: { target: { value: string } }) =>
        setForm({
            email: e.target.value,
            password: form.password,
        })

    const handlePasswordChange = (e: { target: { value: string } }) =>
        setForm({
            email: form.email,
            password: e.target.value,
        })

    const loginHandler = () => {
        console.log(form)
        login(form.email, form.password).then(({ data }) => {
            localStorage.setItem('refresh', data.refresh)
            navigate('/main')
        })
    }

    return (
        <Flex w="100%" h="100%" justifyContent="center">
            <VStack pt="150px">
                <Field.Root>
                    <Field.Label>Логин</Field.Label>
                    <Input type="email" w="400px" onChange={handleLoginChange}></Input>
                    <Field.Label>Пароль</Field.Label>
                    <Input type="password" onChange={handlePasswordChange}></Input>
                </Field.Root>

                <Button w="200px" onClick={loginHandler}>
                    Авторизация
                </Button>
            </VStack>
        </Flex>
    )
}

export default LoginPage
