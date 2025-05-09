import { useMatch, useNavigate } from 'react-router-dom'

import { Button, Flex, VStack, Text } from '@chakra-ui/react'
import { Icon as Iconify } from '@iconify/react'

const menuItems = [
  {
    label: 'Главная',
    icon: 'mdi:home',
    path: '/main',
  },
  {
    label: 'Датасеты',
    icon: 'mdi:file-document',
    path: '/datasets',
  },
  {
    label: 'Обучение',
    icon: 'mdi:brain',
    path: '/training',
  },
  {
    label: 'Пайплайн',
    icon: 'mdi:excavator',
    path: '/pipeline',
  },
  {
    label: 'Обработка',
    icon: 'mdi:transfer',
    path: '/processings',
  },
];

const Menu = () => {
  const navigate = useNavigate()

  const isLogin = useMatch('/')

  if (isLogin) return null

  return (
    <Flex height="100vh" width="160px" bg="gray.200">
      <VStack pt="25px">
        {menuItems.map((item) => (
          <Button
            key={item.label}
            width="160px"
            variant="ghost"
            justifyContent="flex-start"
            onClick={() => navigate(item.path)}
            gap="10px"
          >
            <Iconify icon={item.icon} width="24px" height="24px" />
            <Text>{item.label}</Text>
          </Button>
        ))}
      </VStack>
    </Flex>
  )
}

export default Menu
