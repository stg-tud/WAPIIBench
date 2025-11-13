// This tests complex objects as array elements, only the first element is tested, attribute type is wrong
const axios = require('axios');
axios.request({
    url: 'https://petstore.swagger.io/v2/customers',
    method: 'post',
    data:{
    name: 'name',
    addresses: [
        {
            city: "Darmstadt",
            house:{
                number: "2",
                street: "street"
            }
        }
    ]
}});