// Archive the deals with IDs 001, 002, and 003 as a batch.
const axios = require('axios');
axios.request({
  url:'https://petstore.swagger.io/v2/crm/v3/objects/deals/batch/archive',
  method:'post',
  data:{
    inputs: [
      {id: "001"},
      {id: "002"},
      {id: "003"}
    ]
  }
});