import { useNavigate } from 'react-router-dom'

import { Button, Flex, Image, VStack } from '@chakra-ui/react'

import statsIcon from '../../assets/icons/IoBarChartOutline.svg'
import terminalIcon from '../../assets/icons/IoTerminalOutline.svg'
import deployIcon from '../../assets/icons/BsFillDiagram3Fill.svg'

const Menu = () => {
  const navigate = useNavigate()
  return (
    <Flex height="100vh" width="170px" bg="gray.200" pl="10px">
      <VStack pt="25px">
        <Button
          width="150px"
          variant="ghost"
          justifyContent="flex-start"
          onClick={() => navigate('/')}
        >
          <Image src={statsIcon} pr="10px" />
          Главная
        </Button>
        <Button
          width="150px"
          variant="ghost"
          justifyContent="flex-start"
          onClick={() => navigate('/training')}
        >
          <Image src={terminalIcon} pr="10px" />
          Обучение
        </Button>
        <Button
          width="150px"
          variant="ghost"
          justifyContent="flex-start"
          onClick={() => navigate('/pipeline')}
        >
          <Image src={deployIcon} pr="10px" />
          Деплой
        </Button>
      </VStack>
    </Flex>
  )
}

export default Menu
