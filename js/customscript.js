


/*Hide or unhide white_banner*/
onscroll =  function()
{
  var scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
  
  if (scrollTop>150)
    {  
    /*document.getElementByClassName("white_banner").style.display = "none";*/
      ////document.getElementsByClassName('white_banner').item(0).style.display = "none";
      ////document.getElementsByClassName('white_banner').item(1).style.display = "none";
      document.getElementsByClassName('logo_circle').item(0).classList.remove('hidden-xs');
      document.getElementsByClassName('logo_circle').item(0).classList.remove('hidden-lg');
       document.getElementsByClassName('logo_circle').item(0).classList.remove('hidden-md');
      document.getElementsByClassName('logo_circle').item(0).classList.remove('hidden-sm');
      document.getElementsByClassName('logo_circle').item(0).classList.add('visible-lg-block');
      
      //document.getElementsByClassName('top_banner').item(0).style.display = "block";
      
     

  }
      else
    {
      //document.getElementsByClassName('top_banner').item(0).style.display = "none";
      /*document.getElementByClassName("white_banner").style.display = "block";*/
      ////document.getElementsByClassName('white_banner').item(0).style.display = "block";
            document.getElementsByClassName('logo_circle').item(0).classList.add('hidden-xs');
      document.getElementsByClassName('logo_circle').item(0).classList.add('hidden-lg');
       document.getElementsByClassName('logo_circle').item(0).classList.add('hidden-md');
      document.getElementsByClassName('logo_circle').item(0).classList.add('hidden-sm');

      ////document.getElementsByClassName('white_banner').item(1).style.display = "block";
    
      }


 


    

        }




